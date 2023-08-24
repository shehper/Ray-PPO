#!/bin/sh
#SBATCH --partition=main           # Partition (job queue)
#SBATCH --requeue                   # Return job to the queue if preempted
#SBATCH --job-name=procgen-cleanrl         # Assign a short name to your job
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 8      # cores requested
#SBATCH --mem=1000  # memory in Mb
#SBATCH --output=slurm.%j.out  # send stdout to outfile
#SBATCH --error=slurm.%j.err  # send stderr to errfile
#SBATCH -t 24:00:00  # time requested in hour:minute:second

source ./env/bin/activate
python src/cleanrl_ppo_procgen.py --track=True
deactivate

