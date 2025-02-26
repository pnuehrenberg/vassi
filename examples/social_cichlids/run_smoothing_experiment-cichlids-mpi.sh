#!/bin/bash -l
#
# Python multiprocessing example job script for MPCDF Raven.
#
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
#SBATCH -D ./
#SBATCH -J PYTHON_MP
#SBATCH --nodes=20            # request as many nodes as iterations
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=72    # assign all the cores to that first task to make room for Python's multiprocessing tasks
#SBATCH --time=24:00:00

module purge
module load gcc/10 impi/2021.2

source activate autoscore

srun python3 ./smoothing_experiment-cichlids.py
