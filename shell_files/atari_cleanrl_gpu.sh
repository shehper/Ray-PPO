#!/bin/bash
#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=atari_cleanrl_gpu       # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=8        # Cores per task (>1 if multithread tasks)
#SBATCH --mem=3000                # Real memory (RAM) required (MB)
#SBATCH --time=01:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --output=./slurm.gpu.%N.%j.out
#SBATCH --error=./slurm.gpu.%N.%j.err
#SBATCH --gres=gpu:1

source ./env/bin/activate
python src/ppo_atari.py --track=True
deactivate