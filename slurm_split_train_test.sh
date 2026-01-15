#!/bin/bash
#SBATCH --job-name=Split_train_test
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=16 # 32 tasks per node
#SBATCH --time=24:00:00 # time limits: 1 hour
#SBATCH --error=slurm_logs/split_train_test.err # standard error file
#SBATCH --output=slurm_logs/split_train_test.out # standard output file
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

module load python
source /home/doan/projects/ibm_scripts/torch210/bin/activate
python split_train_test_data.py
deactivate
