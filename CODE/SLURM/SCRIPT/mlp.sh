#!/bin/bash

#SBATCH --job-name=mlp                         
#SBATCH --partition=cpu_long                             
#SBATCH --nodes=1                                 
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

###################################################### DOCUMENTATION ######################################################

# Description: MLP Aggregated/Binned Hawkes Process estimation task
# Usage: sbatch mlp.sh
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###########################################################################################################################

# Checked if folder existed, if not created it
[ ! -d ../OUTPUT ] && mkdir ../OUTPUT

# Moved output/error files in another folder
mv *.out ../OUTPUT

# Checked if necessary modules was loaded, if not, cleaned and did it
{module list 2>&1 | grep -q "anaconda3/2022.10/gcc-11.2.0" && module list 2>&1 | grep -q "cuda/12.0.0/gcc-11.2.0";} || \
{module purge && module load anaconda3/2022.10/gcc-11.2.0 && module load cuda/12.0.0/gcc-11.2.0;}

# Checked if environment was activated, if not, activated it
$CONDA_DEFAULT_ENV | grep -qw hawkes || source activate hawkes

# Run python script (Copy/Paste in .bashrc: export DL="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/CODE/DL")
python "$DL/mlp.py"
