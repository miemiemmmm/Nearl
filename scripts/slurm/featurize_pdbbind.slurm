#!/bin/bash -l
#SBATCH --job-name=FeaturizeTrain                 # Correct the job name
#SBATCH --output=/Matter/training_logs/FeaturizeTrain%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/FeaturizeTrain%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3%4

export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# pdblist=/MieT5/BetaPose/data/train_refined.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/static_pdbbind_refined/


# pdblist=/MieT5/BetaPose/data/test_refined.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/static_pdbbind_refined/


# pdblist=/MieT5/BetaPose/data/train_general.txt
# prefix="train"
# outdir=/Matter/nearl_training_data/static_pdbbind_general/


# pdblist=/MieT5/BetaPose/data/test_general.txt
# prefix="test"
# outdir=/Matter/nearl_training_data/static_pdbbind_general/

# Perform the test
pdblist=/MieT5/BetaPose/data/train_general.txt
prefix="train"
outdir="/Matter/tests/test_new_cutoff/"

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

python /MieT5/BetaPose/scripts/featurize_pdbbind.py \
  -f ${pdblist} \
  --output_dir ${outdir} \
  -t /MieT5/DataSets/PDBBind_complexes/{}_complex.pdb \
  --h5prefix  ${prefix} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} 

