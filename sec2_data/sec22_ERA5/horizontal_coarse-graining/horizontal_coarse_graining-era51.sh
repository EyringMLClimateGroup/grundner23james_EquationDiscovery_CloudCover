#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179

#SBATCH --partition=gpu         # Specify partition name
#SBATCH --time=08:00:00         # Set a limit on the total run time
#SBATCH --nodes=1               # I think one could try more than one node
#SBATCH --mem=950G              # Limit N to 11 (need 80GB per process)          
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
    # Read filename
    file_name=`echo $file | cut -d "." -f1`
    
    # First day of every quarter
    if [[ ( $file_name =~ "-01-01_" || $file_name =~ "-04-01_" || $file_name =~ "-07-01_" || $file_name =~ "-10-01_" ) && ( $file_name =~ "133" || $file_name =~ "246" || $file_name =~ "247" || $file_name =~ "248" ) ]]; then
        if ! [[ -f ${outpath}/${file_name}_R02B05.nc ]] # Do not overwrite
            then
                # Add resolution to the end of the filename
                # Create placeholder file as quickly as possible (to avoid that two processes are writing the same file)
                touch ${outpath}/${file_name}_R02B05.nc
                echo ${file_name}_R02B05.nc
                
                # I think we can run 36 processes per compute node                
                # For the N320 grid use -setgridtype,regular or -setgridtype,regularnn (reduced Gaussian -> Gaussian (linear trafo) -> Unstructured ICON):
                cdo -f nc -P 36 remapcon,${target_grid} -setgridtype,regular ${inpath}/${file} ${outpath}/${file_name}_R02B05.nc
                
        fi
    fi
    
    # First day of every quarter
    if [[ ( $file_name =~ "-01-01_" || $file_name =~ "-04-01_" || $file_name =~ "-07-01_" || $file_name =~ "-10-01_" ) && ( $file_name =~ "129" || $file_name =~ "130" || $file_name =~ "131" || $file_name =~ "132" || $file_name =~ "152" ) ]]; then
        if ! [[ -f ${outpath}/${file_name}_R02B05.nc ]] # Do not overwrite
            then
                # Add resolution to the end of the filename
                # Create placeholder file as quickly as possible (to avoid that two processes are writing the same file)
                touch ${outpath}/${file_name}_R02B05.nc
                echo ${file_name}_R02B05.nc
                
                # For the T639 grid with its spectral coefficients use (spectral -> Gaussian grid point -> Unstructured ICON):
                cdo -f nc -P 36 remapcon,${target_grid} -sp2gp ${inpath}/${file} ${outpath}/${file_name}_R02B05.nc
        fi
    fi
}

# E1ml00_1H_2000-01-01_133.grb
# E1ml00_1H_2000-04-01_133.grb
# E1ml00_1H_2000-07-01_133.grb
# E1ml00_1H_2000-10-01_133.grb

## Main
# inpath='/work/ka1081/DYAMOND/ICON-2.5km'

vars='129 130 131 132 133 152 246 247 248'

for var in $vars; do
    # Used to be /pool/data/ERA5/ml00_1H
    inpath=/pool/data/ERA5/E1/ml/an/1H/${var}
    # inpath=/pool/data/ERA5/E5/ml/an/1H/${var}
    outpath='/work/bd1179/b309170/ERA5/hcg_data'

    target_grid="/pool/data/ICON/grids/public/mpim/0019/icon_grid_0019_R02B05_G.nc"

    files="`ls $inpath`"

    # No more than N processes run at the same time
    N=11
    open_sem $N
    for file in $files; do
        run_with_lock foo $file
    done 
done

# Wait until all threads are finished
wait 

# # Move files into appropriate folders
# mv $outpath/*133* $outpath/q/
# mv $outpath/*246* $outpath/clwc/
# mv $outpath/*247* $outpath/ciwc/
# mv $outpath/*248* $outpath/cc/
# mv $outpath/*130* $outpath/t/
# mv $outpath/*131* $outpath/u/
# mv $outpath/*132* $outpath/v/