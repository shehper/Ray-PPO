#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=tensorboard          # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=2            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=4000                 # Real memory (RAM) required (MB)
#SBATCH --time=02:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out     # STDOUT output file
#SBATCH --error=slurm.%N.%j.err      # STDERR output file (optional)

# mkdir /scratch/$USER/XDGRUNTIMEDIR
# export XDGRUNTIMEDIR=/scratch/$USER/XDGRUNTIMEDIR
# export XDGRUNTIMEDIR=$HOME/tmp   ## needed for jupyter writting temporary files
# srun jupyter notebook --no-browser --ip=0.0.0.0 --port=8889

source ./env/bin/activate

# echo "Starting ${logdir} on port ${port}."

# tensorboard --logdir=/scratch/mas1107/rlalgo --port=16006

srun tensorboard --logdir=/scratch/mas1107/rlalgo --host=0.0.0.0 --port=8889