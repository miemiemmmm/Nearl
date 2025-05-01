#!/bin/bash -l
#SBATCH --job-name=misato_feat_tr                                   # TODO: Correct the job name
#SBATCH --output=/Weiss/benchmark_generation/misato_feat_%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Weiss/benchmark_generation/misato_feat_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0

# TODO: manually set the task count and task id for testing
export SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-6}


echo "Total number of tasks: ${SLURM_ARRAY_TASK_COUNT}"
echo "The task index of this job: ${SLURM_ARRAY_TASK_ID}"

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

prefix="benchmark"

# pdblist=/MieT5/Nearl/data/misato_train.txt    # TODO: here is the test dataset
# outdir="/Weiss/nearl_misato_train"

# pdblist=/MieT5/Nearl/data/misato_test.txt     # TODO: here is the misato test dataset
# outdir="/Weiss/nearl_misato_test"

# pdblist="/MieT5/Nearl/data/casf2016_test.txt"   # TODO: here is the misato test dataset
# outdir="/Weiss/nearl_casf_test"


# pdblist=/MieT5/Nearl/data/misato_train.txt 
# outdir="/Weiss/nearl_misato_train"

# pdblist="/MieT5/Nearl/data/casf2016_test.txt"
# outdir="/Weiss/nearl_misato_test"

pdblist=/MieT5/Nearl/data/misato_train.txt 
outdir="/Weiss/benchmark_generation"


################################################################################
[ ! -d ${outdir} ] && mkdir -p ${outdir}
rm ${outdir}/Output*h5
misatodir="/Matter/misato_database/"
# misatodir="/MieT5/DataSets/misato_database/"
export PYTHONUNBUFFERED=1


# TODO examine with dimension, length, and cutoff 

python3 /MieT5/Nearl/scripts/benchmark_misatofeat.py \
  --pdbcodes    ${pdblist} \
  --dimension   32 \
  --length      24 \
  --cutoff      5.0 \
  --sigma       1.5 \
  --windowsize  20 \
  --misato_dir  ${misatodir} \
  --h5prefix    ${prefix} \
  --output_dir  ${outdir} \
  --task_nr     ${SLURM_ARRAY_TASK_COUNT} \
  --task_index  ${SLURM_ARRAY_TASK_ID} 

realpath ${outdir}/*h5 > ${outdir}/$(echo ${prefix} | cut -c1-2).txt


# python3 /MieT5/Nearl/scripts/benchmark_misatofeat.py -f /MieT5/Nearl/data/casf2016_test.txt -o /tmp/ -t 1 -d 32 -m /Matter/misato_database/ -c 2.5 -s 1.5 