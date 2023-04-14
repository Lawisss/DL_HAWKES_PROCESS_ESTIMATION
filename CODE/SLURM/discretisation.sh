#!/bin/bash

#SBATCH --job-name=hello_numpy
#SBATCH --output=%x.o%j 
#SBATCH --time=00:20:00 
#SBATCH --nodes=<nnodes>
#SBATCH --ntasks-per-node=<ntpn>
#SBATCH --ntasks=1 
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=<ntpt>
#SBATCH --gres=gpu:<ngpus>
#SBATCH --mail-user=firstname.lastname@mywebserver.com 
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

# Load necessary modules
module purge
module load anaconda3/2022.10/gcc-11.2.0 

# Activate anaconda environment
source activate numpy-env

# Run python script
python hello_numpy.py
