#!/bin/bash

#SBATCH --job-name=hyperparameters                          
#SBATCH --partition=cpu_short                               
#SBATCH --nodes=<nnodes>                                    
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=<ntpn>
#SBATCH --cpus-per-task=<ntpt>
#SBATCH --mem=4G
#SBATCH --gres=gpu:<ngpus>
#SBATCH --time=00:20:00

#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

################################################# DOCUMENTATION ###################################################

# Description: Hawkes process hyper-parameters generation task
# Usage: sbatch hyperparameters.sh
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###################################################################################################################

# Loaded necessary modules
module purge
module load anaconda3/2022.10/gcc-11.2.0 

# Activated anaconda environment
source activate numpy-env

# Run python script
python hyperparameters.py
