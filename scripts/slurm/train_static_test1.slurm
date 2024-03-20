#!/bin/bash -l
#SBATCH --job-name=nearl_test1                                # TODO check the name of the job
#SBATCH --output=/Matter/nearl_results/nearl_test1_%a.out     # TODO check the jobname and log dir
#SBATCH --error=/Matter/nearl_results/nearl_test1_%a.err
#SBATCH --array=0-3%2                                         # TODO check the array range and max number of tasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12                                    # TODO check the number of cpus

source /home/yzhang/mamba/bin/loadmamba
micromamba activate pointnet_torch
export PYTHONUNBUFFERED=1


testfiles="/Matter/nearl_dual_static/rad_test.txt%/Matter/nearl_single_static/rad_test.txt%/Matter/nearl_dual_static/mass_test.txt%/Matter/nearl_single_static/mass_test.txt"
trainfiles="/Matter/nearl_dual_static/rad_train.txt%/Matter/nearl_single_static/rad_train.txt%/Matter/nearl_dual_static/mass_train.txt%/Matter/nearl_single_static/mass_train.txt"
outdirs="/Matter/nearl_results/test1_dual_rad%/Matter/nearl_results/test1_single_rad%/Matter/nearl_results/test1_dual_mass%/Matter/nearl_results/test1_single_mass"
datasettypes="dual%single%dual%single"


train_file=$(python3 -c "import sys; print(sys.argv[1].strip().strip('%').split('%')[int(sys.argv[2])])" ${trainfiles} ${SLURM_ARRAY_TASK_ID})
test_file=$(python3 -c "import sys; print(sys.argv[1].strip().strip('%').split('%')[int(sys.argv[2])])" ${testfiles} ${SLURM_ARRAY_TASK_ID})
outdir=$(python3 -c "import sys; print(sys.argv[1].strip().strip('%').split('%')[int(sys.argv[2])])" ${outdirs} ${SLURM_ARRAY_TASK_ID})

pretrained_model=""
start_epoch=0								# NOTE IMPORTANT

total_epoch=40								# NOTE IMPORTANT
batch_size=512								# NOTE IMPORTANT
worker_nr=${SLURM_CPUS_PER_TASK}  # NOTE IMPORTANT

learning_rate=0.001
dataset=$(python3 -c "import sys; print(sys.argv[1].strip().strip('%').split('%')[int(sys.argv[2])])" ${datasettypes} ${SLURM_ARRAY_TASK_ID}) # NOTE IMPORTANT
eval_interval=1000  			   	# NOTE IMPORTANT

echo "This task works on ${dataset} dataset, with ${train_file} as training data, ${test_file} as testing data, writing to ${outdir}."

echo "Task starts at $(date +%Y-%m-%d-%H:%M:%S)"

python /MieT5/BetaPose/scripts/training/train_voxnet_test1.py \
    -train ${train_file} \
    -test  ${test_file} \
    -o 		 ${outdir} \
    \
    --pretrained "${pretrained_model}" \
    --start_epoch ${start_epoch} \
    \
    --data_workers ${worker_nr} \
    --epochs ${total_epoch} \
    --batch_size ${batch_size} \
    \
    -lr ${learning_rate} \
    --interval ${eval_interval} \
    --dataset ${dataset}

echo "Task finished at $(date +%Y-%m-%d-%H:%M:%S)"




