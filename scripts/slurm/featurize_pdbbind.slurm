#!/bin/bash -l
#SBATCH --job-name=FeaturizeTest                               # Correct the job name
#SBATCH --output=/Matter/training_logs/FeaturizeTest%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/FeaturizeTest%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3%4

export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}


# CASF2016 test set: 266 entries
# pdblist=/MieT5/BetaPose/data/casf2016_test.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/static_casf

# Misato test set: 1357 entries 
# pdblist=/MieT5/BetaPose/data/misato_test.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/static_misato_test


# KDeep training set: 3759
pdblist=/MieT5/BetaPose/data/kdeep_train.txt
prefix="train"
outdir="/Matter/nearl_training_data/static_kdeep_trainset"

# Refined training set: 4162
# pdblist=/MieT5/BetaPose/data/refined_train.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/static_refined_train

# Misato training set: 12804
# pdblist=/MieT5/BetaPose/data/misato_train.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/static_train_misato


# Test an arbitrary set 
# pdblist=/MieT5/BetaPose/data/train_general.txt 
# prefix="train" 
# outdir="/Matter/tests/test_new_cutoff/" 

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

python /MieT5/BetaPose/scripts/featurize_pdbbind.py \
  -f ${pdblist} \
  --output_dir ${outdir} \
  -t /MieT5/DataSets/PDBBind_complexes/{}_complex.pdb \
  --h5prefix  ${prefix} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} 

