#!/bin/bash -l
#SBATCH --job-name=VoxNet_Single
#SBATCH --output=/diskssd/yzhang/FEater_data/results_single_vox/VoxNet_Single.out
#SBATCH --error=/diskssd/yzhang/FEater_data/results_single_vox/VoxNet_Single.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24



source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch


"/Matter/nearl_dual_static/index_label.csv"
datafile = "/Matter/nearl_dual_static/voxel.h5"; 


python /MieT5/Nearl/scripts/training/train_voxnet_test1.py \
  --data_path "/Matter/nearl_single_static/voxel.h5" \
  --dataset "dual" \
  --cuda 1 \



