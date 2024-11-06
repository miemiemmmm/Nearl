#!/bin/bash -l
#SBATCH --job-name=DynaProp                                # TODO: Correct the job name
#SBATCH --output=/Matter/training_logs/DynaProp_%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Matter/training_logs/DynaProp_%a.err      # TODO: Correct the log folder and file name 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-14%4                                        # TODO: Match the desired concurrent tasks
#SBATCH --cpus-per-task=6                                    # TODO: Match the cpu number with desired concurrent tasks

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-12}    # if running in non-SLURM environment, set the number of CPUs here
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-19}    # if running in non-SLURM environment, set the task ID here

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

# tasklist="/MieT5/BetaPose/scripts/slurm/static_tasks.csv"   # array=0-11%4                # Experiment: 3D baseline on static data
# tasklist="/MieT5/BetaPose/scripts/slurm/static_tasks_h.csv"   # array=0-8%4               # Experiment: Hilbert baseline on static data
# tasklist="/MieT5/BetaPose/scripts/slurm/ensemble_tasks.csv"  # array=0-5%2                # Experiment: 3D baseline on refined ensemble data
# tasklist="/MieT5/BetaPose/scripts/slurm/ensemble_tasks_2.csv"  # array=0-5%4              # Experiment: 3D baseline on misato ensemble data
# tasklist="/MieT5/BetaPose/scripts/slurm/ensemble_tasks_h.csv"  # array=0-3%4              # Experiment: Hilbert baseline on ensemble data
# tasklist="/MieT5/BetaPose/scripts/slurm/ensemble_prop_exploration.csv"  # array=0-38%4    # Experiment: property exploration (3D) on ensemble data
# tasklist="/MieT5/BetaPose/scripts/slurm/ensemble_prop_exploration_h.csv"  # array=0-25%4  # Experiment: property exploration (Hilbert) on ensemble data
tasklist="/MieT5/BetaPose/scripts/slurm/dynamic_prop_exploration.csv"      # array=0-14%4   # Experiment: property exploration (3D) on pure dynamic features


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

# train_data="/Matter/nearl_training_data/ensemble_refined_train/tr.txt"
# test_data="/Matter/nearl_training_data/ensemble_casf/te.txt"
# target_datatags="static_C_lig%static_C_prot%static_N_lig%static_N_prot%static_O_lig%static_O_prot%static_S_lig%static_S_prot"
# "hb_acceptor_prot%hb_donor_prot%aromatic_prot%hb_acceptor_lig%hb_donor_lig%aromatic_lig%static_C_prot%static_N_prot%static_O_prot%static_S_prot%static_C_lig%static_N_lig%static_O_lig%static_S_lig%"
# labeltag="label_pcdt"
# output_dir="/tmp/testmodel"
# model="voxnet"

# Slowly learning something: pafnucy, atom3d, deeprank, kdeep (Requires lower learning rate at 1e-4)
# Yes: resnet3d, voxnet, gnina2017, gnina2018

# echo "$model, $train_data, $test_data, $target_datatags, $labeltag, $output_dir"

python /MieT5/BetaPose/scripts/train_models.py \
  --model ${model} --optimizer adam --loss-function mse \
  --training_data ${train_data} --test_data ${test_data} --output_folder ${output_dir}  \
  --epochs 100 --batch_size 64  --test_number 1280 --lr-init 0.001 --lr-decay-steps 10 --lr-decay-rate 0.5 --output_dimension 1 \
  --tags ${target_datatags} --labeltag ${labeltag} --data_workers ${SLURM_CPUS_PER_TASK} ${more_options} 
  
