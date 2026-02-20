#!/bin/bash
#SBATCH --job-name=run_oracle_vs_avg_ens
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=16 # 32 tasks per node
#SBATCH --time=24:00:00 # time limits: 1 hour
#SBATCH --error=slurm_logs/run_oracle_vs_avg_ens.err # standard error file
#SBATCH --output=slurm_logs/run_oracle_vs_avg_ens.out # standard output file
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gprod_gssi # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

module load python
source /home/doan/projects/ibm_scripts/torch210/bin/activate
python run_oracle.py
python run_ave_ens.py
python merge_scores_without_selectors.py
deactivate
