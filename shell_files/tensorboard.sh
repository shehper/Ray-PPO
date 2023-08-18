#!/bin/bash
#SBATCH --job-name="tensorboard"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source ./env/bin/activate

echo "Starting ${logdir} on port ${port}."

tensorboard --logdir=$logdir 