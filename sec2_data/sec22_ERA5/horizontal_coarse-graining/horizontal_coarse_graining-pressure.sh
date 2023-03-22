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
    
    # Takes the coarse-grained lnsp as input
    # if [[ $file =~ "2016-01" && $file =~ "152" ]]; then
    if [[ $file =~ "152" ]]; then
        file_name=`echo "$file" | awk -F'152' '{print $1}'`
        
        if ! [[ -f ${outpath}/pa/${file_name}pa_R02B05.nc ]] # Do not overwrite
            then
                # Add resolution to the end of the filename
                # Create placeholder file as quickly as possible (to avoid that two processes are writing the same file)
                touch ${outpath}/pa/${file_name}pa_R02B05.nc
                echo ${file_name}pa_R02B05.nc
                
                # Compute y = exp(x)
                cdo exp ${outpath}/${file_name}152_R02B05.nc ${outpath}/${file_name}sp_R02B05.nc
                # Compute a + b*y and create 137 nc-files (one for each level)
                vlevel=0
                while read line
                do
                   a=`echo $line | cut -d ',' -f2`
                   b=`echo $line | cut -d ',' -f3`
                   if [ "$vlevel" != "0" ]; then
                      cdo setlevel,${vlevel} -addc,$a -mulc,$b ${outpath}/${file_name}sp_R02B05.nc ${outpath}/${file_name}pa_R02B05_${vlevel}.nc
                   fi
                   vlevel=$((${vlevel}+1))
                done < /home/b/b309170/bd1179_work/ERA5/processing/l137_vgrid.csv
                # Piece together those 137 nc-files
                cdo merge ${outpath}/${file_name}pa_R02B05_?.nc ${outpath}/${file_name}pa_R02B05_merged_1.nc 
                cdo merge ${outpath}/${file_name}pa_R02B05_??.nc ${outpath}/${file_name}pa_R02B05_merged_2.nc 
                cdo merge ${outpath}/${file_name}pa_R02B05_???.nc ${outpath}/${file_name}pa_R02B05_merged_3.nc 
                cdo merge ${outpath}/${file_name}pa_R02B05_merged_?.nc ${outpath}/${file_name}pa_R02B05_lnsp.nc 
                
                # Rename lnsp to pa
                cdo setpartabn,/work/bd1179/b309170/ERA5/processing/lnsp_to_pressure,convert ${outpath}/${file_name}pa_R02B05_lnsp.nc ${outpath}/pa/${file_name}pa_R02B05.nc
                
                # Done!
                \rm ${outpath}/${file_name}sp_R02B05.nc 
                \rm ${outpath}/${file_name}lnsp_R02B05.nc
                \rm ${outpath}/${file_name}pa_R02B05_*.nc 
        fi
    fi
}

## Main: Run after horizontal_coarse_graining.sh!!
# inpath='/work/ka1081/DYAMOND/ICON-2.5km'
module load cdo

files="`ls $outpath`"
outpath='/work/bd1179/b309170/ERA5/hcg_data'

# No more than N processes run at the same time
N=11
open_sem $N
for file in $files; do
    run_with_lock foo $file
done 

# Wait until all threads are finished
wait 