#!/bin/bash -l
#SBATCH --job-name=TrainMisato                 # Correct the job name
#SBATCH --output=/Matter/nearl_result3/TrainMisato_%a.out     #  TODO:Correct the log folder and file name 
#SBATCH --error=/Matter/nearl_result3/TrainMisato_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=9-16%4
#SBATCH --cpus-per-task=4          # Match with the cpu number in the script

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-6}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-6}

source /home/yzhang/mamba/bin/loadmamba
micromamba activate nearl_dev

tasklist="/MieT5/BetaPose/data/test_tasks.csv"  

model=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][0])")
train_data=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][1])")
test_data=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][2])")
echo "Model is ${model}; Training data is ${train_data}; Test data is ${test_data}; "

target_datatags=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][3])")
labeltag=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][4])")
output_dir=$(python -c "import pandas as pd; print(pd.read_csv('${tasklist}', index_col=False, sep=' ', header=None).loc[${SLURM_ARRAY_TASK_ID}][5])")
echo "Working on ${target_datatags} with label ${labeltag} and output to ${output_dir}"

more_options=""

python /MieT5/BetaPose/scripts/training/train_models.py \
  --model ${model} --optimizer adam --loss-function mse \
  --training_data ${train_data} --test_data ${test_data} --output_folder ${output_dir} --output_dimension 1 \
  --epochs 40 --batch_size 64  --test_number 1000 --lr-init 0.001 --lr-decay-steps 20 --lr-decay-rate 0.5 \
  --data_workers  ${SLURM_CPUS_PER_TASK}  --augment 1 \
  --tags ${target_datatags} --labeltag ${labeltag} ${more_options} \
  --production 1
