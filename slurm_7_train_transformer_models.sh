#!/bin/bash
#SBATCH --job-name=train_transformer_model
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=16 # 32 tasks per node
#SBATCH --time=24:00:00 # time limits: 1 hour
#SBATCH --error=slurm_logs/train_transformer_model.err # standard error file
#SBATCH --output=slurm_logs/train_transformer_model.out # standard output file
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gprod # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

module load python
source /home/doan/projects/ibm_scripts/torch210/bin/activate
python train_transformer_model.py
deactivate
