# DL & Hawkes Process Estimation

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<!--- Results illustration here --->

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contact](#contact)
- [Acknowledgment](#Acknowledgment)
- [License](#license)

## Description

* Mixed unsupervised-supervised model to leverage sentimental and non-sentimental information from corpus. 
* Learned word vectors that capture document semantic and sentimental information. 
* Evaluated on a movie reviews dataset from the Internet Movie Database (IMDB). 
* Outperformed unsupervised vector sentiment classification approaches that usually fail to capture sentiments.

<!--- Project features here --->

## Getting Started

* Simulated hyperparameters, binned HP according to parameters (horizon time, kernel function, baseline). 
* Statistical inference using MLP regressor on parameters {Branching ratio: $eta$, baseline intensity: $mu$}. 
* Evaluated our approach on a large dataset of informal movie reviews to compare it with published results.

### Prerequisites

To download required packages:

```sh
pip install -r requirements.txt
```

```sh
conda env create -f environment.yml
```

To download dataset:

```sh
Error. Dataset not available.
```

### Installation

1. Clone the [project](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION) with Git.

```sh
git clone https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION.git
```
2. Open [tutorial.ipynb](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/tutorial.ipynb) in your browser.

```sh
python -m tutorial.ipynb
```

## Usage

* To run the code: [tutorial.ipynb](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/tutorial.ipynb). This file allows you to load python files and results.
* To get details about model architecture, training and visualization, see [report](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/REPORT/report.pdf).
* To leverage preprocessing, AHP/MLP/VAE parameters and results, use global [variable](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/tree/main/CODE/VARIABLES) files.
* To deepen incomplete pytest unit tests, check files in [test](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/tree/main/TEST) folder.

|                                         File                                                                       |               Extension               |               Folder                  |               Complete                |               Function                |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| [Tutorial](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/tutorial.ipynb)     | .ipynb                                   | CODE                  	     | ✔️                                   | Executed project sections
| [Hyperparameters](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES/hyperparameters.py)     | .py                                   | HAWKES                  	     | ✔️                                   | Computed HP hyperparameters
| [Hawkes](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES/hawkes.py)   | .py                                   | HAWKES                    	     | ✔️                                   | Simulated HP
| [Discretisation](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES/discretisation.py)                     | .py                                   | HAWKES                           | ✔️                                   | Discretized HP                       |
| [Dataset](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/PREPROCESSING/dataset.py)     | .py                                   | PREPROCESSING           		     | ✔️                                   | Preprocessed HP
| [MLP](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/DL/mlp.py)       | .py                                   | DL                         | ✔️                                   | Executed MLP model                |
| [VAE](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/DL/vae.py)    | .py                                   | DL                        | ✔️                                   | Executed VAE model               |
| [Hawkes_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/hawkes_var.py)				 | .py                                 | VARIABLES                       	     | ✔️                                   | Monitored HP variables          |
| [Preprocessing_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/preprocessing_var.py)     | .py                                   | VARIABLES                  	     | ✔️                                   | Monitored preprocessing variables	                  |
| [MLP_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/mlp_var.py)   | .py                                   | VARIABLES                    	     | ✔️                                   | Monitored MLP variables	                  |
[VAE_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/vae_var.py)   | .py                                   | VARIABLES                    	     | ✔️                                   | Monitored VAE variables	                  |
[Evaluation_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/evaluation_var.py)   | .py                                   | VARIABLES                 	     | ✔️                                   | Monitored evaluation variables	                  |
[Utils](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/UTILS/utils.py)   | .py                                   | UTILS                    	     | ✔️                                   | Executed saving/loading                  |
[Utils_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/UTILS/utils_mpi.py)   | .py                                   | UTILS                    	     | ❌                                  | Executed MPI saving/loading	                  |
[Hyperparameters_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES_MPI/hyperparameters_mpi.py)   | .py                                   | HAWKES_MPI                   	     | ❌                                   | Computed MPI HP hyperparameters       |
[Hawkes_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES_MPI/hawkes_mpi.py)   | .py                                   | HAWKES_MPI                    	     | ❌                                   | Simulated MPI HP	                  |
[Discretisation_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES_MPI/discretisation_mpi.py)   | .py                                   | HAWKES_MPI                    	     | ❌                                   | Discretized MPI HP	                  |
[Config](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/config.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Installed conda environment
[Hyperparameters](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/hyperparameters.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Executed MPI hyperparameters
[Hawkes](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/hawkes.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Executed MPI HP
[Discretisation](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/discretisation.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Executed MPI discretisation
[MLP](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/mlp.sh)   | .sh                                   | SLURM                    	     | ❌                                   | Executed MPI MLP
[VAE](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/vae.sh)   | .sh                                   | SLURM                    	     | ❌                                   | Executed MPI VAE

## Roadmap

* Leveraged document-level sentiment annotations abundant online (consumer reviews for movies or products). 
* Described our dataset ([Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)) and our model. 
* Evaluated our approach on a large dataset of informal movie reviews to compare it with published results.

## Contact

* Nicolas Girard - nico.girard22@gmail.com.

## Acknowledgment

* Tom Keane - Imperial College London - [Statistical Inference for Hawkes Processes with Deep Learning](https://tom-keane.github.io/project_1.pdf).
* Shlomovich et al. - Imperial College London - [Parameter Estimation of Binned Hawkes Processes](https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2050247).

## License

<a href="https://choosealicense.com/licenses/mit/"><img src="https://raw.githubusercontent.com/johnturner4004/readme-generator/master/src/components/assets/images/mit.svg" height=40 />MIT License</a>
