#!/bin/bash -l
#SBATCH --job-name=FeaturizeMisatoTest                 # Correct the job name
#SBATCH --output=/Matter/nearl_training_data/test_jobs/FeaturizeMisatoTest%a.out     # Correct the log folder and file name 
#SBATCH --error=/Matter/nearl_training_data/test_jobs/FeaturizeMisatoTest%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5%3

echo "Total number of tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "The task index of this job: ${SLURM_ARRAY_TASK_ID}"

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

python3 /MieT5/BetaPose/scripts/featurize_misato.py \
  --task_nr ${SLURM_ARRAY_TASK_COUNT} \
  --task_index ${SLURM_ARRAY_TASK_ID}


