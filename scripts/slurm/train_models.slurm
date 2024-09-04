#!/bin/bash -l
#SBATCH --job-name=StaticBaseline                                # TODO: Correct the job name
#SBATCH --output=/Matter/training_logs/StaticBaseline_%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/StaticBaseline_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-11%4                                       # TODO: Match the desired concurrent tasks
#SBATCH --cpus-per-task=6                                   # TODO: Match the cpu number with desired concurrent tasks

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-12}    # if running in non-SLURM environment, set the number of CPUs here
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}    # if running in non-SLURM environment, set the task ID here

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

tasklist="/MieT5/BetaPose/scripts/slurm/static_tasks.csv"   # array=0-11%4
# tasklist="/MieT5/BetaPose/data/static_tasks_augment.csv"
# tasklist="/MieT5/BetaPose/data/static_tasks_hilb.csv"  # array=0-11%2

# tasklist="/MieT5/BetaPose/data/property_exploration.csv"  # array=0-23%4
# tasklist="/MieT5/BetaPose/data/property_exploration_hilb.csv"  # array=0-11%2


# tasklist="/Matter/tests/test_new_cutoff/task.csv"   # TODO: Only for test purpose

model=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][0])")
train_data=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][1])")
test_data=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][2])")
echo "Model is ${model}; Training data is ${train_data}; Test data is ${test_data}; "

target_datatags=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][3])")
labeltag=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][4])")
output_dir=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][5])")
echo "Working on ${target_datatags} with label ${labeltag} and output to ${output_dir}" 

more_options="--augment 0 --production 1"

# train_data="/Matter/nearl_training_data/static_refined_train/tr.txt"
# test_data="/Matter/nearl_training_data/static_misato_test/te.txt"
# target_datatags="hb_acceptor_prot%hb_donor_prot%aromatic_prot%hb_acceptor_lig%hb_donor_lig%aromatic_lig%static_C_prot%static_N_prot%static_O_prot%static_S_prot%static_C_lig%static_N_lig%static_O_lig%static_S_lig%"
# labeltag="pk_original"
# output_dir="/tmp/testmodel"
# model="voxnet"


# Slowly learning something: pafnucy, atom3d, deeprank, kdeep (Requires lower learning rate at 1e-4)
# Yes: resnet3d, voxnet, gnina2017, gnina2018, 


python /MieT5/BetaPose/scripts/train_models.py \
  --model ${model} --optimizer adam --loss-function mse \
  --training_data ${train_data} --test_data ${test_data} --output_folder ${output_dir}  \
  --epochs 100 --batch_size 64  --test_number 1280 --lr-init 0.001 --lr-decay-steps 10 --lr-decay-rate 0.5 --output_dimension 1 \
  --tags ${target_datatags} --labeltag ${labeltag} --data_workers  ${SLURM_CPUS_PER_TASK} ${more_options} 
  
