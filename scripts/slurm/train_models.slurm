#!/bin/bash -l
#SBATCH --job-name=rebuttal_trainings                                # TODO: Correct the job name
#SBATCH --output=/Weiss/nearl_rebuttal/rebuttal_trainings_%a.out     # TODO: Correct the log folder and file name 
#SBATCH --error=/Weiss/nearl_rebuttal/rebuttal_trainings_%a.err      # TODO: Correct the log folder and file name 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-20%4                                       # TODO: Match the desired concurrent tasks
#SBATCH --cpus-per-task=6                                    # TODO: Match the cpu number with desired concurrent tasks

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}    # if running in non-SLURM environment, set the number of CPUs here
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}    # if running in non-SLURM environment, set the task ID here

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

# tasklist="/MieT5/Nearl/scripts/slurm/static_tasks.csv"   # array=0-11%4                # Experiment: 3D baseline on static data
# tasklist="/MieT5/Nearl/scripts/slurm/static_tasks_h.csv"   # array=0-8%4               # Experiment: Hilbert baseline on static data
# tasklist="/MieT5/Nearl/scripts/slurm/ensemble_tasks.csv"  # array=0-5%2                # Experiment: 3D baseline on refined ensemble data
# tasklist="/MieT5/Nearl/scripts/slurm/ensemble_tasks_2.csv"  # array=0-5%4              # Experiment: 3D baseline on misato ensemble data
# tasklist="/MieT5/Nearl/scripts/slurm/ensemble_tasks_h.csv"  # array=0-3%4              # Experiment: Hilbert baseline on ensemble data
# tasklist="/MieT5/Nearl/scripts/slurm/ensemble_prop_exploration.csv"  # array=0-38%4    # Experiment: property exploration (3D) on ensemble data
# tasklist="/MieT5/Nearl/scripts/slurm/ensemble_prop_exploration_h.csv"  # array=0-25%4  # Experiment: property exploration (Hilbert) on ensemble data
# tasklist="/MieT5/Nearl/scripts/slurm/dynamic_prop_exploration.csv"      # array=0-14%4   # Experiment: property exploration (3D) on pure dynamic features


tasklist="/MieT5/Nearl/scripts/slurm/tasks_rebuttal.csv"


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

# Slowly learning something: pafnucy, atom3d, deeprank, kdeep (Requires lower learning rate at 1e-4)
# Yes: resnet3d, voxnet, gnina2017, gnina2018

python /MieT5/Nearl/scripts/train_models.py \
  --model ${model} --optimizer adam --loss-function mse \
  --training_data ${train_data} --test_data ${test_data} --output_folder ${output_dir}  \
  --epochs 100 --batch_size 64  --test_number 1280 --lr-init 0.001 --lr-decay-steps 10 --lr-decay-rate 0.5 --output_dimension 1 \
  --tags ${target_datatags} --labeltag ${labeltag} --data_workers ${SLURM_CPUS_PER_TASK} ${more_options} 
  
