# DL & Hawkes Process Estimation

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<!--- Results illustration here --->

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contacts](#contacts)
- [Acknowledgments](#Acknowledgments)

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

To download dataset:

```sh
tar -xf aclImdb_v1.tar.gz
```

### Installation

1. Clone the [project](https://github.com/MoonMess/NLP_Project.git) with Git.

```sh
git clone https://github.com/MoonMess/NLP_Project.git
```
2. Open [visualization.ipynb](https://github.com/MoonMess/NLP_Project/blob/main/visualization.ipynb) in your browser.

```sh
python -m visualization.ipynb
```
## Usage

## Usage

* To run the code: [preprocessing.ipynb](https://github.com/Lawisss/G10_IRS_MARKET_TREND/blob/main/PREPROCESSING/preprocessing.ipynb). This file allows you to load python files and results.
* To get details about model architecture, training and the visualization, contacted the project author directly.
* To leverage regression SOTA/ML models results, most of them are not available for confidentiality reasons.

|                                         File                                                                       |               Extension               |               Folder                  |               Complete                |               Function                |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| [Preprocessing](https://github.com/Lawisss/EUREX_EGB_PRICE_IMPACT/blob/main/LIBRARY/preprocessing.py)     | .py                                   | LIBRARY                  	     | ✔️                                   | Preprocessed EUREX EGB metaorders	                  |
| [Kyle](https://github.com/Lawisss/EUREX_EGB_PRICE_IMPACT/blob/main/LIBRARY/kyle.py)   | .py                                   | LIBRARY                    	     | ✔️                                   | Executed Kyle model	                  |
| [Sqrt_root_law](https://github.com/Lawisss/EUREX_EGB_PRICE_IMPACT/blob/main/LIBRARY/sqrt_root_law.py)                     | .py                                   | LIBRARY                           | ✔️                                   | Executed Square Root Law model                       |
| [Istar](https://github.com/Lawisss/EUREX_EGB_PRICE_IMPACT/blob/main/LIBRARY/istar.py)     | .py                                   | LIBRARY           		     | ❌                                   | Executed I-STAR model                   |
| [Automl](https://github.com/Lawisss/EUREX_EGB_PRICE_IMPACT/blob/main/LIBRARY/automl.py)       | .py                                   | LIBRARY                         | ✔️                                   | Executed AutoML model                |
| [Autodl](https://github.com/Lawisss/EUREX_EGB_PRICE_IMPACT/blob/main/LIBRARY/autodl.py)    | .py                                   | LIBRARY                        | ❌                                   | Executed AutoDL model               |
| [Evaluation](https://github.com/Lawisss/EUREX_EGB_PRICE_IMPACT/blob/main/LIBRARY/evaluation.py)				  | .py                                 | LIBRARY                       	     | ✔️                                   | Executed/Monitored evaluation          |




## Contacts

* Nicolas Girard - nico.girard22@gmail.com.

## License

<a href="https://choosealicense.com/licenses/mit/"><img src="https://raw.githubusercontent.com/johnturner4004/readme-generator/master/src/components/assets/images/mit.svg" height=40 />MIT License</a>
