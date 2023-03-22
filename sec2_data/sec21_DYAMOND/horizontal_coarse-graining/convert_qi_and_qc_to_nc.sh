#!/bin/bash
#=============================================================================
#SBATCH --account=bd1179

#SBATCH --partition=compute    # Specify partition name
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --nodes=5
#SBATCH --output=/pf/b/b309170/slurm_scripts/%j.out

#SBATCH --job-name=convert_to_nc.sh

#=============================================================================

# Needs 1100s per file and node

inpath='/work/ka1081/DYAMOND/ICON-2.5km'
outpath='/work/bd1179/b309170/DYAMOND'

files="`ls $inpath`"

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
    inpath='/work/ka1081/DYAMOND/ICON-2.5km'
    outpath='/work/bd1179/b309170/DYAMOND'
    
    file_name=`echo $file | cut -d "." -f1`
    if [[ $file_name =~ "tot_qc_dia" || $file_name =~ "tot_qi_dia" || $file_name =~ "clw" || $file_name =~ "cli" ]]; then
        if [[ ! -f $outpath/$file_name.nc ]]; then
            cdo -P 24 -f nc copy $inpath/$file $outpath/$file_name.nc
        fi
    fi
}

# We have 128 cores per compute node
N=5
open_sem $N
for file in $files; do
    run_with_lock foo $file
done