#!/bin/bash -l
#SBATCH --job-name=Data_Featurize
#SBATCH --output=/users/yzhang/autoScripts/featurization_test.out
#SBATCH --error=/users/yzhang/autoScripts/featurization_test.err
#SBATCH --constraint=mc
#SBATCH --account=uzh6
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-core=1

echo ">>>> Job Start at: $(date '+%d.%m.%Y %H.%M.%S') <<<<"

# Find conda and setup conda environment
which conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlenv


echo -e "\nMemory Information"
free -h
echo -e "\nCPU Information:"
lscpu
echo -e "\n"

srun python /users/yzhang/scratch/featurization/BetaPose/scripts/prepare_pdbbind.py  &

pid=$!

while true; do
  echo "############################################################################"
  echo "Resource Usage: "
  sstat --format=JobID,AveCPU,AvePages,AveRSS,AveVMSize -j $SLURM_JOB_ID --noheader 1>&2
  top -b -n 1 | grep "Cpu\|MiB" 1>&2
  sleep 60
  if ! ps -p $pid > /dev/null ; then
    break
  fi
  echo "############################################################################"
done