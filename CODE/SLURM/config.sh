#!/bin/bash

#SBATCH --job-name=configuration                          
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

###################################################### DOCUMENTATION ######################################################

# Description: Conda environment configuration task
# Usage: sbatch "$HOME/$SLURM/configuration.sh" 
# (In .bashrc: export SLURM="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/SLURM")
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###########################################################################################################################

# Setup conda env, ensured your .conda directory is located on your workir, and move it if not
[ -L ~/.conda ] && unlink ~/.conda
[ -d ~/.conda ] && mv -v ~/.conda $WORKDIR
[ ! -d $WORKDIR/.conda ] && mkdir $WORKDIR/.conda
ln -s $WORKDIR/.conda ~/.conda

# Cleaned and loaded necessary modules
module purge
module load anaconda3/2022.10/gcc-11.2.0 

# Created conda environment
conda create --file=environment.yml --force

# Asked if you want to activate environment
while true; do
    read -p "Do you want to activate "hawkes" environment ? [y/n or Y/N]:" answer
    case $answer in
        [Yy]* )
            # Activated environment
            source activate hawkes
            break;;
        [Nn]* )
            # Exit script
            exit;;
        * )
            # Repeat question
            echo "Answer "Y" or "N" (case-insensitive).";;
    esac
done
