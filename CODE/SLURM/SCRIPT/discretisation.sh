#!/bin/bash

#SBATCH --job-name=discretize                         
#SBATCH --partition=cpu_short                               
#SBATCH --nodes=1                                
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
##SBATCH --gres=gpu:1 (uncomment to use it)
#SBATCH --time=00:10:00

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

################################################# DOCUMENTATION ###################################################

# Description: Hawkes process aggregation task
# Usage: sbatch "$SLURM/discretisation.sh" 
# (Copy/Paste in .bashrc: export SLURM="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/SLURM/SCRIPT")
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###################################################################################################################

# Checked if folder existed, if not created it
[ ! -d ../OUTPUT ] && mkdir ../OUTPUT

# Set output/error files in folder
#SBATCH --output=OUTPUT/%x_%j.out
#SBATCH --error=OUTPUT/%x_%j.out

# Checked if necessary modules was loaded, if not, cleaned and did it
{module list 2>&1 | grep -q anaconda3/2022.10/gcc-11.2.0;} || {module purge && module load anaconda3/2022.10/gcc-11.2.0;}

# Activated conda environment if existed, else created and activated it
{source activate hawkes 2>/dev/null;} || {conda create --name hawkes --file=environment.yml --force && source activate hawkes;}

# Run python script (Copy/Paste in .bashrc: export HAWKES="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/HAWKES")
python $HAWKES/discretisation.py"
