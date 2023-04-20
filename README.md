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

* Leveraged document-level sentiment annotations abundant online (consumer reviews for movies or products). 
* Described our dataset ([Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)) and our model. 
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
| [Tutorial](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/tutorial.ipynb)     | .ipynb                                   | CODE                  	     | ✔️                                   | Preprocessed EUREX EGB metaorders	
| [Hyperparameters](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES/hyperparameters.py)     | .py                                   | HAWKES                  	     | ✔️                                   | Preprocessed EUREX EGB metaorders	                  |
| [Hawkes](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES/hawkes.py)   | .py                                   | HAWKES                    	     | ✔️                                   | Executed Kyle model	                  |
| [Discretisation](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES/discretisation.py)                     | .py                                   | HAWKES                           | ✔️                                   | Executed Square Root Law model                       |
| [Dataset](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/PREPROCESSING/dataset.py)     | .py                                   | PREPROCESSING           		     | ✔️                                   | Executed I-STAR model                   |
| [MLP](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/DL/mlp.py)       | .py                                   | DL                         | ✔️                                   | Executed AutoML model                |
| [VAE](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/DL/vae.py)    | .py                                   | DL                        | ✔️                                   | Executed AutoDL model               |
| [Hawkes_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/hawkes_var.py)				 | .py                                 | VARIABLES                       	     | ✔️                                   | Executed/Monitored evaluation          |
| [Preprocessing_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/preprocessing_var.py)     | .py                                   | VARIABLES                  	     | ✔️                                   | Preprocessed EUREX EGB metaorders	                  |
| [MLP_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/mlp_var.py)   | .py                                   | VARIABLES                    	     | ✔️                                   | Executed Kyle model	                  |
[VAE_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/vae_var.py)   | .py                                   | VARIABLES                    	     | ✔️                                   | Executed Kyle model	                  |
[Evaluation_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/VARIABLES/evaluation_var.py)   | .py                                   | VARIABLES                 	     | ✔️                                   | Executed Kyle model	                  |
[Utils](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/UTILS/utils.py)   | .py                                   | UTILS                    	     | ✔️                                   | Executed Kyle model	                  |
[Utils_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/UTILS/utils_mpi.py)   | .py                                   | UTILS                    	     | ❌                                  | Executed Kyle model	                  |
[Hyperparameters_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES_MPI/hyperparameters_mpi.py)   | .py                                   | HAWKES_MPI                   	     | ❌                                   | Executed Kyle model	                  |
[Hawkes_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES_MPI/hawkes_mpi.py)   | .py                                   | HAWKES_MPI                    	     | ❌                                   | Executed Kyle model	                  |
[Discretisation_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/HAWKES_MPI/discretisation_mpi.py)   | .py                                   | HAWKES_MPI                    	     | ❌                                   | Executed Kyle model	                  |
[Config](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/config.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Executed Kyle model
[Hyperparameters](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/hyperparameters.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Executed Kyle model	
[Hawkes](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/hawkes.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Executed Kyle model	
[Discretisation](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/discretisation.sh)   | .sh                                   | SLURM                    	     | ✔️                                   | Executed Kyle model	
[MLP](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/mlp.sh)   | .sh                                   | SLURM                    	     | ❌                                   | Executed Kyle model	
[VAE](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/CODE/SLURM/vae.sh)   | .sh                                   | SLURM                    	     | ❌                                   | Executed Kyle model	

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
