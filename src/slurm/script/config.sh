#!/bin/bash

#SBATCH --job-name=config                          
#SBATCH --partition=cpu_short                               
#SBATCH --nodes=1                                
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

###################################################### DOCUMENTATION ######################################################

# Description: Conda environment configuration task
# Usage: sbatch config.sh
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###########################################################################################################################

# Checked if folder existed, if not created it
[ ! -d ../output ] && mkdir ../output

# Moved output/error files in another folder
mv *.out ../output

# Setup conda environment, ensured .conda directory is located on workir, if not moved it
[ -L ~/.conda ] && unlink ~/.conda
[ -d ~/.conda ] && mv -v ~/.conda $WORKDIR
[ ! -d $WORKDIR/.conda ] && mkdir $WORKDIR/.conda
ln -s $WORKDIR/.conda ~/.conda

# Cleaned and loaded necessary modules
module purge && module load anaconda3/2022.10/gcc-11.2.0

# Checked if environment already existed, if not, created it. Finally, activated it
# (Copy/Paste in .bashrc: export VENV="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/environment.yml")
conda env list | grep -qw hawkes && source activate hawkes || conda env create --file=$VENV --force && source activate hawkes