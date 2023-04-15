#!/bin/bash

#SBATCH --job-name=discretisation                          
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

# Description: Hawkes process aggregation task
# Usage: sbatch discretisation.sh
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###################################################################################################################

# Cleaned and moaded necessary modules
module purge
module load anaconda3/2022.10/gcc-11.2.0 

# Activated anaconda environment
source activate hawkes

# Run python script
python "$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/HAWKES/discretisation.py"
