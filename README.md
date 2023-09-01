# DL & Hawkes Process Estimation

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<!--- Results illustration here --->

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contact](#contact)
- [Acknowledgment](#acknowledgment)
- [License](#license)

## Description

- Estimated binned Hawkes processes using variational bayesian methods and deep learning in Python (Pytorch).
- Simulated and predicted baseline intensity, self-exciting and intensity decay rate using NN (CUDA, MPI).
- Inferred observation distributions and conditional intensity leveraging unsupervised technique (Autoencoders).
- Extended methods to Hawkes processes with more complex dimensions, excitation kernels and HPC architecture.

<!--- Project features here --->

## Getting Started

- Simulated hyperparameters and binned HP based on parameters (horizon time, kernel function, baseline).
- Estimated using MLP and LSTM regressor the binned HP parameters (Branching ratio: $\eta$, Baseline intensity: $\mu$).
- Inferred using Poisson-VAE (dueling decoder) the joint distribution of ${{\eta,\mu}}$ and the conditional intensity $\lambda$.
- Assessed and compared model estimations and errors according to binned HP parameters ($\Delta$, $E$, $\eta$, $\beta$).

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

2. Open [tutorial.ipynb](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/tutorial.ipynb) in your browser.

```sh
python -m tutorial.ipynb
```

## Usage

- To run the code: [tutorial.ipynb](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/tutorial.ipynb). This file allows you to load python files and results.
- To get details about model architecture, training, testing and visualization, see [report](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/report/report.pdf).
- To leverage preprocessing, AHP/MLP/LSTM/VAE parameters and results, use global [variable](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/tree/main/src/variables) files.
- To deepen incomplete pytest unit tests, check files in [test](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/tree/main/test) folder.

|                                         File                                                                       |               Extension               |               Folder                  |               Complete                |               Function                |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| [tutorial](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/tutorial.ipynb)     | .ipynb                                   | src                        | ✔️                                   | Project demonstration
| [app](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/app.py)     | .ipynb                                   | src                        | ✔️                                   | Gathered project sections
| [run](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/run.py)     | .ipynb                                   | src                        | ✔️                                   | Executed CLI app
| [hyperparameters](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/hawkes/hyperparameters.py)     | .py                                   | hawkes                        | ✔️                                   | Computed HP hyperparameters
| [hawkes](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/hawkes/simulation.py)   | .py                                   | hawkes                          | ✔️                                   | Simulated HP
| [discretisation](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/hawkes/discretisation.py)                     | .py                                   | hawkes                           | ✔️                                   | Discretized HP                       |
| [dataset](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/preprocessing/dataset.py)     | .py                                   | preprocessing               | ✔️                                   | Preprocessed HP
| [linear_model](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/dl/linear_model.py)    | .py                                   | dl                        | ✔️                                   | Executed Linear model               |
| [mlp_model](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/dl/mlp_model.py)       | .py                                   | dl                         | ✔️                                   | Executed MLP model                |
| [lstm_model](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/dl/lstm_model.py)       | .py                                   | dl                         | ✔️                                   | Executed LSTM model                |
| [vae_model](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/dl/vae_model.py)    | .py                                   | dl                        | ✔️                                   | Executed Poisson-VAE model               |
| [dueling_decoder](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/dl/dueling_decoder.py)    | .py                                   | dl                        | ✔️                                   | Executed Dueling Decoder model               |
[eval](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/evaluation/eval.py)   | .py                                   | evaluation                     | ✔️                                   | Errors evaluation
[error_viz](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/visualization/error_viz.py)   | .py                                   | visualization                     | ✔️                                   | Errors visualization
| [hawkes_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/variables/hawkes_var.py)     | .py                                 | variables                           | ✔️                                   | Monitored HP variables          |
| [prep_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/variables/prep_var.py)     | .py                                   | variables                     | ✔️                                   | Monitored preprocessing variables   |
[parser_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/variables/parser_var.py)   | .py                                   | variables                     | ✔️                                   | Monitored parser variables |
| [mlp_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/variables/mlp_var.py)   | .py                                   | variables                        | ✔️                                   | Monitored MLP variables                   |
| [lstm_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/variables/lstm_var.py)   | .py                                   | variables                        | ✔️                                   | Monitored LSTM variables                   |
[vae_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/variables/vae_var.py)   | .py                                   | variables                        | ✔️                                   | Monitored Poisson-VAE variables                   |
[eval_var](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/variables/eval_var.py)   | .py                                   | variables                     | ✔️                                   | Monitored evaluation variables               |
[utils](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/utils/utils.py)   | .py                                   | utils                        | ✔️                                   | Executed saving/loading                  |
[utils_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/utils/utils_mpi.py)   | .py                                   | utils                        | ❌                                  | Executed MPI saving/loading                   |
[hyperparameters_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/hawkes_mpi/hyperparameters_mpi.py)   | .py                                   | hawkes_mpi                      | ❌                                   | Computed MPI HP hyperparameters       |
[hawkes_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/hawkes_mpi/simulation_mpi.py)   | .py                                   | hawkes_mpi                      | ❌                                   | Simulated MPI HP                   |
[discretisation_mpi](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/hawkes_mpi/discretisation_mpi.py)   | .py                                   | hawkes_mpi                       | ❌                                   | Discretized MPI HP                   |
[config](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/slurm/script/config.sh)   | .sh                                   | slurm                        | ✔️                                   | Installed conda environment
[run](https://github.com/Lawisss/DL_HAWKES_PROCESS_ESTIMATION/blob/main/src/slurm/script/run.sh)   | .sh                                   | slurm                        | ✔️                                   | Executed project main

## Roadmap

- Refined and improved work on MLE, MLP and LSTM models (Methods, Fine-tuning, Optimization).
- Explored the extension of the horizon $T$ and other parameter ranges (extreme values or extended intervals).
- Extended approaches to new machine learning techniques (self-attention mechanisms, transformers).

## Contact

- Nicolas Girard - nico.girard22@gmail.com.

## Acknowledgment

- Tom Keane - Imperial College London - 2020 - [Statistical Inference for Hawkes Processes with Deep Learning](https://tom-keane.github.io/project_1.pdf).
- Shlomovich et al. - Imperial College London - 2022 - [Parameter Estimation of Binned Hawkes Processes](https://www.tandfonline.com/doi/full/10.1080/10618600.2022.2050247).

## License

<a href="https://choosealicense.com/licenses/mit/"><img src="https://raw.githubusercontent.com/johnturner4004/readme-generator/master/src/components/assets/images/mit.svg" height=40 />MIT License</a>
