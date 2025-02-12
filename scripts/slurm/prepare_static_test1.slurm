#!/bin/bash -l
#SBATCH --job-name=testjob                 # Correct the job name
#SBATCH --output=/Matter/nearl_results/testjob.out     # Correct the log folder and file name 
#SBATCH --error=/Matter/nearl_results/testjob.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-26%4
#SBATCH --cpus-per-task=4                  # Match with the cpu number in the script


source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch

datafile="/Weiss/FEater_Dual_PDBHDF/TrainingSet_Dataset.h5"; 
outfolder="/Matter/nearl_dual_static/"

# datafile="/Weiss/FEater_Single_PDBHDF/TrainingSet_Dataset.h5";
# outfolder="/Matter/nearl_single_static"


python /MieT5/Nearl/scripts/residue_static_generation.py \
  -f ${datafile} -o ${outfolder}    \
  -b 500 -c ${SLURM_CPUS_PER_TASK}  \
  --weight_type mass \


