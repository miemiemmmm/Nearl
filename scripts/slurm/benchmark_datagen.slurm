#!/bin/bash -l
#SBATCH --job-name=benchmark_datagen                                # Correct the job name
#SBATCH --output=/Matter/training_logs/benchmark_datagen_%a.out     # Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/benchmark_datagen_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-3%1

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

echo "Executing the task ${SLURM_ARRAY_TASK_ID}" 

python3 /MieT5/Nearl/scripts/benchmark_datagen.py -f /MieT5/trajlist.txt -t ${SLURM_ARRAY_TASK_ID} -c 8 -s 20 --force 1 --num-repeat 10

