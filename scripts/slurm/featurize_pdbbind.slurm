#!/bin/bash -l
#SBATCH --job-name=static_misato_trainset_dim32                               # Correct the job name
#SBATCH --output=/Matter/training_logs/static_misato_trainset_dim32%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/static_misato_trainset_dim32%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-3%4

# Set the default values if not in slurm environment
export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-6}


# CASF2016 test set: 266 entries
# pdblist=/MieT5/BetaPose/data/casf2016_test.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/static_casf_dim32

# Misato test set: 1357 entries 
# pdblist=/MieT5/BetaPose/data/misato_test.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/static_misato_test_dim32


# KDeep training set: 3759
# pdblist=/MieT5/BetaPose/data/kdeep_train.txt
# prefix="train"
# outdir="/Matter/nearl_training_data/static_kdeep_trainset"

# Refined training set: 4162
# pdblist=/MieT5/BetaPose/data/refined_train.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/static_refined_train_dim32

# Misato training set: 12804
# pdblist=/MieT5/BetaPose/data/misato_train.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/static_misato_trainset_dim32


# Test an arbitrary set 
# pdblist=/MieT5/BetaPose/data/train_general.txt 
# prefix="train" 
# outdir="/Matter/tests/test_new_cutoff/" 

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

python /MieT5/BetaPose/scripts/featurize_pdbbind.py \
  -f ${pdblist} \
  --dimension 32 \
  --length 20 \
  --output_dir ${outdir} \
  -t /MieT5/DataSets/PDBBind_complexes/{}_complex.pdb \
  --h5prefix  ${prefix} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} \
  --cpu_nr ${SLURM_CPUS_PER_TASK} 

