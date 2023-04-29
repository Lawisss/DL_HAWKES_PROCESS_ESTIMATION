#!/bin/bash

#SBATCH --job-name=hawkes                         
#SBATCH --partition=cpu_long                              
#SBATCH --nodes=1                                    
#SBATCH --ntasks=1                                          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
##SBATCH --gres=gpu:1 (uncomment to use it)
#SBATCH --time=168:00:00

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

#SBATCH --mail-user=nicolas.girard@centralesupelec.fr   
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH --propagate=NONE

################################################# DOCUMENTATION ###################################################

# Description: Hawkes process simulation task
# Usage: sbatch hawkes.sh
# Params: Check documentation: https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/06_slurm_jobs_management/

###################################################################################################################

# Checked if folder existed, if not created it
[ ! -d ../output ] && mkdir ../output

# Moved output/error files in another folder
mv *.out ../output

# Checked if necessary modules was loaded, if not, cleaned and did it
module list 2>&1 | grep -q anaconda3/2022.10/gcc-11.2.0 || module purge && module load anaconda3/2022.10/gcc-11.2.0

# Checked if environment was activated, if not, activated it
$CONDA_DEFAULT_ENV | grep -qw hawkes || source activate hawkes

# Run python script (Copy/Paste in .bashrc: export HAWKES="$HOME/Documents/VAE_HAWKES_PROCESS_ESTIMATION/src/hawkes")
python $HAWKES/hawkes.py hawkes_simulations "hawkes_simulations_ruche.parquet"