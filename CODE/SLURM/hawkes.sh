#!/bin/bash

#SBATCH --job-name=hawkes                         
#SBATCH --partition=cpu_short                               
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
##SBATCH --gres=gpu:1 (uncomment to use it)
#SBATCH --time=00:10:00

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

################################################# DOCUMENTATION ###################################################

# Description: Hawkes process simulation task
# Usage: sbatch "$SLURM/hawkes.sh"
# (In .bashrc: export SLURM="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/SLURM")
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###################################################################################################################

# Checked if necessary modules was loaded, if not, cleaned and did it
if ! module list | grep -q anaconda3/2022.10/gcc-11.2.0; then
  module purge
  module load anaconda3/2022.10/gcc-11.2.0
fi

# Checked if virtual environment is activated, if not, did it
if ! conda info --envs | grep -q "^hawkes "; then
  # Activated environment
  source activate hawkes
fi

# Run python script (In .bashrc: export HAWKES="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/HAWKES")
python "$HAWKES/hawkes.py"
