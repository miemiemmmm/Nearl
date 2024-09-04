#!/bin/bash -l
#SBATCH --job-name=FeaturizeMisato_multistatic                 # Correct the job name
#SBATCH --output=/Matter/training_logs/FeaturizeMisato_multistatic_%a.out     # Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/FeaturizeMisato_multistatic_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3%4

# TODO: manually set the task count and task id for testing
export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Total number of tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "The task index of this job: ${SLURM_ARRAY_TASK_ID}"
source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

# outdir="/Matter/nearl_training_data/testjob3"                  # TODO Correct the output directory before running
# pdbcodes="/Matter/nearl_training_data/testjob3/noprot.txt"     # TODO Correct the pdbcode file before running


#### Misato dataset for general set 
outdir=/Matter/nearl_training_data/ensemble_casf
prefix="test"
pdbcodes=/MieT5/BetaPose/data/casf2016_test.txt


# outdir=/Matter/nearl_training_data/misato_multi_static_props
# prefix="train"
# pdbcodes=/MieT5/BetaPose/data/train_general.txt



#### Misato dataset for refined set
# outdir=/Matter/nearl_training_data/misato_pdbbind_refined
# prefix="train"
# pdbcodes=/MieT5/BetaPose/data/train_refined.txt

# outdir=/Matter/nearl_training_data/misato_pdbbind_refined
# prefix="test"
# pdbcodes=/MieT5/BetaPose/data/test_refined.txt


################################################################################
[ ! -d ${outdir} ] && mkdir -p ${outdir}
# misatodir="/Matter/misato_database/"
misatodir="/MieT5/DataSets/misato_database/"
python3 /MieT5/BetaPose/scripts/featurize_misato.py \
  --pdbcodes    ${pdbcodes} \
  --misato_dir  ${misatodir} \
  --h5prefix  ${prefix} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} \
  --output_dir  ${outdir}
