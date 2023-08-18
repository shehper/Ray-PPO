#!/bin/sh
#SBATCH --partition=main           # Partition (job queue)
#SBATCH --requeue                   # Return job to the queue if preempted
#SBATCH --job-name=atari-pong         # Assign a short name to your job
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 8      # cores requested
#SBATCH --mem=1000  # memory in Mb
#SBATCH --output=slurm.%j.out  # send stdout to outfile
#SBATCH --error=slurm.%j.err  # send stderr to errfile
#SBATCH -t 24:00:00  # time requested in hour:minute:second

# Number of gpus per node is specified as #SBATCH --gpus-per-node=2

source ../env/bin/activate
python ordinary_ppo_atari.py --track=True
deactivate

