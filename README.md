# DeepRunAway
This repository contains a suite of physics-constrained deep learning surrogate models that describe characteristics pertaining to relativistic "runaway" electrons. Physics Informed Neural Networks (PINNs) [[Raissi 2019](https://doi.org/10.1016/j.jcp.2018.10.045) , [Karniadakis 2021](https://www.nature.com/articles/s42254-021-00314-5)] are primarily used to learn PDE solutions to the kinetic equation for runaway electrons, as well as learning the solution across a range of plasma parameters, enabling the use of the trained model to provide rapid inferences on the solution for optimization and inverse problems.

** INSERT SOME COOL HIGHLIGHT MOVIE OF THE PINN TRAINING **

# Getting started
To get started, Python 3.12 is recommended, and the user can create the virtual environment with Conda ([see the official guide to install conda if needed](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)) or pip only. Unless otherwise stated, all PINN scripts are written in PyTorch.
### For Conda:
The evironment can be created with the [environment.yml](environment.yml) file by executing the command ```conda env create -n <ENV_NAME> -f environment.yml```, where ```<ENV_NAME>``` is the user-defined environment name.

### For pip:
The environment can be created with the [requirements.txt](requirements.txt) file by executing the command ```python -m venv <ENV_NAME> && <ENV_NAME>/bin/pip install -r requirements.txt```, where ```<ENV_NAME>``` is the user-defined environment name.


## Hardware
We recommend GPUs (Nvidia GPUs have only been tested) for training the PINNs, however, some of the smaller PINN models can be trained on modern laptops, such as Apple silicon macbooks (Apple silicon is limited to float 32 precision, so performance will not be as good as float 64, but may be sufficient for certain scenarios. For a in depth analysis, see [this study](https://doi.org/10.48550/arXiv.2501.16371)).

