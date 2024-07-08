#!/bin/bash -l
#SBATCH --job-name=FeaturizeMisato                 # Correct the job name
#SBATCH --output=/Matter/nearl_training_data/test_train/FeaturizeMisato%a.out     # Correct the log folder and file name 
#SBATCH --error=/Matter/nearl_training_data/test_train/FeaturizeMisato%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3%4

# TODO: manually set the task count and task id for testing
# export SLURM_ARRAY_TASK_COUNT=1
# export SLURM_ARRAY_TASK_ID=0

echo "Total number of tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "The task index of this job: ${SLURM_ARRAY_TASK_ID}"
source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

# outdir="/Matter/nearl_training_data/prod_train"
# outdir="/Matter/nearl_training_data/prod_test"
# training_set = "/MieT5/BetaPose/data/misato_train.txt"
# training_set = "/MieT5/BetaPose/data/misato_test.txt"

outdir="/Matter/nearl_training_data/test_train"   # TODO Correct the output directory before running
pdbcodes="/MieT5/BetaPose/data/misato_train.txt"   # TODO Correct the pdbcode file before running

################################################################################
misatodir="/Matter/misato_database/"
python3 /MieT5/BetaPose/scripts/featurize_misato.py \
  --pdbcodes    ${pdbcodes} \
  --misato_dir  ${misatodir} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} \
  --output_dir  ${outdir}
