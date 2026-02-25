#!/bin/bash
#SBATCH --job-name=plot_results_on_test
#SBATCH --nodes=1 # 1 node
#SBATCH --ntasks-per-node=4 # 32 tasks per node
#SBATCH --time=24:00:00 # time limits: 1 hour
#SBATCH --error=slurm_logs/plot_results_on_test.err # standard error file
#SBATCH --output=slurm_logs/plot_results_on_test.out # standard output file
#SBATCH --partition=lprod # partition name
#SBATCH --mail-type=END              # type of event notification
#SBATCH --mail-user=thihoaithu.doan@gssi.it   # mail address

module load python
source /home/doan/projects/ibm_scripts/torch210/bin/activate
python plot_results_current.py
deactivate
