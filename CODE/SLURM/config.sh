#!/bin/bash

#SBATCH --job-name=config                          
#SBATCH --partition=cpu_short                               
#SBATCH --nodes=1                                
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

###################################################### DOCUMENTATION ######################################################

# Description: Conda environment configuration task
# Usage: sbatch "$SLURM/config.sh" 
# (In .bashrc: export SLURM="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/SLURM")
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###########################################################################################################################

# Checked if folder existed, if not created it
[ ! -d OUTPUT ] && mkdir OUTPUT

# Set output/error files in folder
#SBATCH --output=OUTPUT/%x_%j.out
#SBATCH --error=OUTPUT/%x_%j.out

# Setup conda environment, ensured .conda directory is located on workir, if not moved it
[ -L ~/.conda ] && unlink ~/.conda
[ -d ~/.conda ] && mv -v ~/.conda $WORKDIR
[ ! -d $WORKDIR/.conda ] && mkdir $WORKDIR/.conda
ln -s $WORKDIR/.conda ~/.conda

# Cleaned and loaded necessary modules
module purge
module load anaconda3/2022.10/gcc-11.2.0

# Checked if environment already exists
if ! conda info --envs | grep -q "^hawkes "; then
    # Created environment
    conda create --name hawkes --file=environment.yml --force
fi

