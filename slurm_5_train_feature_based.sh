#!/bin/bash
#SBATCH --job-name=train_feature_based
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=16 # 32 tasks per node
#SBATCH --time=24:00:00 # time limits: 1 hour
#SBATCH --error=slurm_logs/train_feature_based.err # standard error file
#SBATCH --output=slurm_logs/train_feature_based.out # standard output file
#SBATCH --gres=gpu:a100
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

module load python
source /home/doan/projects/ibm_scripts/torch210/bin/activate
python train_feature_based.py
deactivate
