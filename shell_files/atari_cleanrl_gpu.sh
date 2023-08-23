#!/bin/bash
#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --exclusive
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=atapri_cleanrl_gpu       # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=8        # Cores per task (>1 if multithread tasks)
#SBATCH --mem=3000                # Real memory (RAM) required (MB)
#SBATCH --time=10:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --output=./slurm.gpu.%N.%j.out
#SBATCH --error=./slurm.gpu.%N.%j.err
#SBATCH --gres=gpu:1
#SBATCH --exclude=cuda[001-008],gpu[005-008],pascal[001-010],volta[001-003]  # blacklist slow nodes

source ./env/bin/activate
python src/ppo_atari.py --track=True
deactivate