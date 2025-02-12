#!/bin/bash -l
#SBATCH --job-name=FeaturizeInHouse                 # Correct the job name
#SBATCH --output=/Matter/nearl_training_data/inhouse75_train/FeaturizeInHouse%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Matter/nearl_training_data/inhouse75_train/FeaturizeInHouse%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3%4

# TODO: manually set the task count and task id for testing
export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Total number of tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "The task index of this job: ${SLURM_ARRAY_TASK_ID}"
source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

# outdir=/Matter/nearl_training_data/inhouse75_test
# pdbcodes=/MieT5/Nearl/data/inhouse_test.txt


outdir=/Matter/nearl_training_data/inhouse75_train
pdbcodes=/MieT5/Nearl/data/inhouse_train.txt

# outdir="/Matter/nearl_training_data/inhouse75_test"
# pdbcodes="/MieT5/Nearl/data/inhouse_test.txt"


################################################################################
python3 /MieT5/Nearl/scripts/featurize_trajectory.py \
  --trajfiles   ${pdbcodes} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} \
  --output_dir  ${outdir}
