#!/bin/bash
#=============================================================================
#SBATCH --account=bd1083

#SBATCH --partition=gpu         # Specify partition name
#SBATCH --time=05:00:00         # Set a limit on the total run time
#SBATCH --nodes=1               # I think one could try more than one node
#SBATCH --mem=450G              # Limit N to 11 (need 80GB per process)          
#SBATCH --output=/home/b/b309170/scratch/slurm_scripts/%j.out
#SBATCH --error=/home/b/b309170/scratch/slurm_scripts/%j.out
#SBATCH --mail-type=END,FAIL

#SBATCH --job-name=hcg.sh

#=============================================================================

# DO NOT RUN MULTIPLE INSTANCES OF THIS JOB IN PARALLEL; THEY WILL INTERFERE

# 90 files in 8 hours (on Mistral)
# We process (no_nodes)*N files concurrently! So 100 files with the current setting.

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo /home/b/b309170/scratch/pipe-$$
    exec 3<>/home/b/b309170/scratch/pipe-$$
    rm /home/b/b309170/scratch/pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

foo () {
    echo $file
    
    # Takes the coarse-grained lnsp, t, q, geopotential as input
    if [[ $file =~ "129" ]]; then
    
        # Construct output file name 
        part1=`echo "$file" | awk -F'129' '{print $1}'`
        out_file_zf=${part1}zf_R02B05.nc
        out_file_zh=${part1}zh_R02B05.nc
        
        if ! [[ -f ${path}/${out_file_zf} ]] # Do not overwrite
            then
                # Add resolution to the end of the filename
                # Create placeholder file as quickly as possible (to avoid that two processes are writing the same file)
                touch ${path}/${out_file_zf}
                touch ${path}/${out_file_zh}
                echo ${out_file_zf}
                echo ${out_file_zh}
                
                # Merge temperature and specific humidity & convert to grib2
                cdo -f grb2 copy [ -merge ${path}/t/${part1}130_R02B05.nc ${path}/q/${part1}133_R02B05.nc ] ${path}/${part1}tq_R02B05.grib
                
                # Merge geopotential and ln(p_s) & convert to grib2
                cdo -f grb2 copy [ -merge ${path}/${part1}129_R02B05.nc ${path}/${part1}152_R02B05.nc ] ${path}/${part1}zlnsp_R02B05.grib
                
                # Compute height on full and half levels for all model levels
                python /home/b/b309170/bd1179_work/ERA5/processing/compute_geopotential_on_ml.py ${path}/${part1}tq_R02B05.grib ${path}/${part1}zlnsp_R02B05.grib -o ${path}/${part1}zf_R02B05.grib
                python /home/b/b309170/bd1179_work/ERA5/processing/compute_geopotential_on_ml_zh.py ${path}/${part1}tq_R02B05.grib ${path}/${part1}zlnsp_R02B05.grib -o ${path}/${part1}zh_R02B05.grib
                
                # Convert the output back to nc
                cdo -f nc setgrid,${target_grid} [ -copy ${path}/${part1}zf_R02B05.grib ] ${path}/geop_${out_file_zf}
                cdo -f nc setgrid,${target_grid} [ -copy ${path}/${part1}zh_R02B05.grib ] ${path}/geop_${out_file_zh}
                
                # Convert from geopotential to geopotential height
                g=9.80665
                cdo divc,$g ${path}/geop_${out_file_zh} ${path}/geop_h_${out_file_zh}
                cdo divc,$g ${path}/geop_${out_file_zf} ${path}/geop_h_${out_file_zf}
                
                # Convert from geopotential height to geometric height
                Re=6371229 # Radius of the earth
                # Re*h
                cdo mulc,${Re} ${path}/geop_h_${out_file_zh} ${path}/geop_h_1_${out_file_zh}
                cdo mulc,${Re} ${path}/geop_h_${out_file_zf} ${path}/geop_h_1_${out_file_zf}
                # Re - h = -(h - Re)
                cdo mulc,-1 -subc,${Re} ${path}/geop_h_${out_file_zh} ${path}/geop_h_2_${out_file_zh}
                cdo mulc,-1 -subc,${Re} ${path}/geop_h_${out_file_zf} ${path}/geop_h_2_${out_file_zf}
                # Re*h/(Re - h)
                cdo div ${path}/geop_h_1_${out_file_zh} ${path}/geop_h_2_${out_file_zh} ${path}/unsorted_${out_file_zh}
                cdo div ${path}/geop_h_1_${out_file_zf} ${path}/geop_h_2_${out_file_zf} ${path}/unsorted_${out_file_zf}
                
                # Actually, the python files break the temporal order. It needs to be fixed. Be careful that input_file != output_file:
                cdo timsort ${path}/unsorted_${out_file_zh} ${path}/${out_file_zh}
                cdo timsort ${path}/unsorted_${out_file_zf} ${path}/${out_file_zf}
                
                # Done!
                \rm ${path}/unsorted_${out_file_zh}
                \rm ${path}/unsorted_${out_file_zf}
                
                \rm ${path}/${part1}zlnsp_R02B05.grib
                \rm ${path}/${part1}zf_R02B05.grib
                \rm ${path}/${part1}zh_R02B05.grib
                
                \rm ${path}/geop_${out_file_zh}
                \rm ${path}/geop_h_${out_file_zh}
                \rm ${path}/geop_h_1_${out_file_zh}
                \rm ${path}/geop_h_2_${out_file_zh}
                
                \rm ${path}/geop_${out_file_zf}
                \rm ${path}/geop_h_${out_file_zf}
                \rm ${path}/geop_h_1_${out_file_zf}
                \rm ${path}/geop_h_2_${out_file_zf}
        fi
    fi
}

# Main: Run after horizontal_coarse_graining.sh!!
. ~/.bashrc
conda activate sympy

path=/work/bd1179/b309170/ERA5/hcg_data

target_grid="/pool/data/ICON/grids/public/mpim/0019/icon_grid_0019_R02B05_G.nc"

files="`ls $path`"

# No more than N processes run at the same time
N=8
open_sem $N
for file in $files; do
    run_with_lock foo $file
done 

# Wait until all threads are finished
wait 