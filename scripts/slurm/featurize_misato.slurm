#!/bin/bash -l
#SBATCH --job-name=ensemble_mistrain                                # Correct the job name
#SBATCH --output=/Matter/training_logs/ensemble_mistrain_%a.out     # Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/ensemble_mistrain_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3%4

# TODO: manually set the task count and task id for testing
export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-6}
export PYTHONUNBUFFERED=1

echo "Total number of tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "The task index of this job: ${SLURM_ARRAY_TASK_ID}"
source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev


# CASF2016 test set: 266 entries
# pdblist=/MieT5/Nearl/data/casf2016_test.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/ensemble_casf

# Misato test set: 1357 entries 
# pdblist=/MieT5/Nearl/data/misato_test.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/ensemble_misato_test

# Refined training set: 4162
# pdblist=/MieT5/Nearl/data/refined_train.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/ensemble_refined_train

# Misato training set: 12804
# pdblist=/MieT5/Nearl/data/misato_train.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/ensemble_misato_train


################################ Dynamic data ################################

# pdblist=/MieT5/Nearl/data/casf2016_test.txt
# prefix="test"
# outdir="/Matter/nearl_training_data/dynamic_casf"

# pdblist=/MieT5/Nearl/data/misato_test.txt
# prefix="test"
# outdir="/Matter/nearl_training_data/dynamic_misato_test"

# pdblist=/MieT5/Nearl/data/refined_train.txt
# prefix="train"
# outdir="/Matter/nearl_training_data/dynamic_refined_train"

pdblist=/MieT5/Nearl/data/misato_train.txt
prefix="train"
outdir="/Matter/nearl_training_data/dynamic_misato_train"

################################################################################
[ ! -d ${outdir} ] && mkdir -p ${outdir}
rm ${outdir}/*h5
# misatodir="/Matter/misato_database/"
misatodir="/MieT5/DataSets/misato_database/"
python3 /MieT5/Nearl/scripts/featurize_misato.py \
  --pdbcodes    ${pdblist} \
  --dimension   32 \
  --length      20 \
  --misato_dir  ${misatodir} \
  --h5prefix    ${prefix} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} \
  --output_dir  ${outdir} 

realpath ${outdir}/*h5 > ${outdir}/$(echo ${prefix} | cut -c1-2).txt
