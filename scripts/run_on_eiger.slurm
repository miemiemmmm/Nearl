#!/bin/bash -l
#SBATCH --job-name=Data_Featurize
#SBATCH --output=/users/yzhang/autoScripts/featurization_test.out
#SBATCH --error=/users/yzhang/autoScripts/featurization_test.err
#SBATCH --constraint=mc
#SBATCH --account=uzh6
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=2

echo ">>>> Job starts at: $(date '+%Y-%m-%d %H:%M:%S') <<<<"

INFO_DIR="/users/yzhang/autoScripts/"
MAMBA_SCRIPT="/scratch/e1000/yzhang/mamba/bin/loadmamba"
ENV_NAME="env2"
SCRIPT_TO_RUN="/scratch/e1000/yzhang/featurization/BetaPose/scripts/prepare_single_residue.py"
TEMP_WORKSPACE="/scratch/e1000/yzhang/test_dask_workspace/"


# Find micromamba and setup micromamba environment
source ${MAMBA_SCRIPT:-/scratch/e1000/yzhang/mamba/bin/loadmamba};
export OMP_NUM_THREADS=2
export PYTHONPATH=""
if [ ${#MAMBA_EXE} -eq 0 ]; then
  exit 0; 
fi 

module load cray/22.05
module load cpeGNU
module load cray-python/3.9.12.1

which micromamba && (echo "Good Found the Micromamba") || (echo "Not found the micromamba executable" && exit 1)
micromamba activate ${ENV_NAME:-mlenv};
micromamba env list

# dask scheduler &

echo -e "\nMemory Information"
free -h
echo -e "\nCPU Information:"
lscpu
echo -e "\n"

# Task parallelization happens within the Python and Dask schuduler/Dask workers
echo -e "#!/bin/bash -l \necho ########################################################TODO HERE \npython /scratch/e1000/yzhang/featurization/BetaPose/scripts/prepare_single_residue.py" > ${INFO_DIR}feat_runner.sh && chmod +x ${INFO_DIR}feat_runner.sh; 
srun ${INFO_DIR}feat_runner.sh &

pid=$!
echo "PID is ${pid}"; 

while true; do
  echo "############################################################################" 1>&2
  echo "Resource Usage: " 1>&2
  sstat --format=JobID,AveCPU,AvePages,AveRSS,AveVMSize -j $SLURM_JOB_ID --noheader 1>&2
  top -b -n 1 | grep "Cpu\|Node\|MiB" 1>&2
  ps -aux | awk -v name="yzhang" 'NR==1 || $1 == name' 1>&2
  echo "############################################################################" 1>&2
  sleep 30
  if ! ps -p $pid > /dev/null ; then
    exit 0
    break
  fi
done
