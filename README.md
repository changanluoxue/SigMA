# SigMA
This work implements "SigMA: Path Signatures and Multi-head Attention for Learning Parameters in fBm-driven SDEs" by Xianglin Wu, Chiheb Ben Hammouda, and Cornelis W. Oosterlee. 

## 🚀 Prerequisites
All experiments were run on a system with CUDA12.8, utilizing GPU acceleration. The specific versions of the key libraries were: PyTorch2.8.0, Python3.10.12. 

## 🏗️ Project Structure
The "example" files correspond to the seven different experiments in the article. 
Specifically, "example1" to "example5" use synthetic data, and "example6" and "example7" use real data. 
The "paths_simulation" file is used to generate the synthetic data for "example1" to "example5". 
The "market_data" folder contains the real data used by "example6" and "example7". 
Within the "example" files, setting __name__ = "__training__" is for training the neural networks, and setting __name__ = "__results__" or __name__ = "__calibration__" is for outputting the experimental results. 
The "layers" and "NNmodels" files together define the neural network model structures required for all experiments. 
The "plots" file is used for generating the figures in the paper. 
The "utils" file defines the trainer functions. 


