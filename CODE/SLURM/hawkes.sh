#!/bin/bash

#SBATCH --job-name=hawkes                         
#SBATCH --partition=cpu_short                               
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

################################################# DOCUMENTATION ###################################################

# Description: Hawkes process simulation task
# Usage: sbatch "$HOME/$SLURM/hawkes.sh"
# (In .bashrc: export SLURM="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/SLURM")
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###################################################################################################################

# Loaded necessary modules
module purge
module load anaconda3/2022.10/gcc-11.2.0 

# Activated anaconda environment
source activate hawkes

# Run python script (In .bashrc: export HAWKES="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/HAWKES")
python "$HOME/$HAWKES/hawkes.py"
