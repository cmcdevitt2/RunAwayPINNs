** UNDER CONSTRUCTION !!! **

# RunAwayPINNs
This repository contains a description of relativistic "runaway" electrons with Physics Informed Neural Networks (PINNs) [[Raissi 2019](https://doi.org/10.1016/j.jcp.2018.10.045) , [Karniadakis 2021](https://www.nature.com/articles/s42254-021-00314-5)]. The respository is comprised of two components, being the PINN surrogates and the general PINN training scripts. Further information is provided as one navigates into either directory.

** INSERT SOME COOL HIGHLIGHT MOVIE OF THE PINN TRAINING **

# Getting started
To get started, Python 3.12 is recommended, and the user can create the virtual environment with Conda ([see the official guide to install conda if needed](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)) or pip only. Unless otherwise stated, all PINN scripts are written in PyTorch.
### For Conda:
The evironment can be created with the [environment.yml](environment.yml) file by executing the command ```conda env create -n <ENV_NAME> -f environment.yml```, where ```<ENV_NAME>``` is the user-defined environment name.

### For pip:
The environment can be created with the [requirements.txt](requirements.txt) file by executing the command ```python -m venv <ENV_NAME> && <ENV_NAME>/bin/pip install -r requirements.txt```, where ```<ENV_NAME>``` is the user-defined environment name.


## Hardware
We recommend GPUs (Nvidia GPUs have only been tested) for training the PINNs, however, some of the smaller PINN models can be trained on modern laptops, such as Apple silicon macbooks (Apple silicon is limited to float 32 precision, so performance will not be as good as float 64, but may be sufficient for certain scenarios. For a in depth analysis, see [this study](https://doi.org/10.48550/arXiv.2501.16371)). A brief comparison of runtimes for training a simple RunAwayPINN where it is a 2D PDE and represents the simplest and least computationally intensive scenario, is given in the table below (as of __,__,2025).

**Simulation setup:**  50,000 training points and 3 hidden layers with 32 neurons per layer, such that the model fits on all hardware. The simulation was ran on both the ADAM optimizer from PyTorch and the SOAP optimizer from the pytorch-optimizer library, which represent first order and second order optimization routines for deep learning, respectively.

| Device   | Precision | ~Runtime (ADAM) | ~Runtime (SOAP) |
| :------ | :------: | :------: | :------: |
| (CPU) [Apple M1 Pro 16GB](https://support.apple.com/en-us/111902)   | Float32   | 12m    |Right    |
| (GPU) [Apple M1 Pro 16GB](https://support.apple.com/en-us/111902)   | Float32   | 3m   |Right    |
| (CPU) [AMD "Turin" EPYC 9655P](https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9655p.html)  | Float64   | 18m   |Right    |
| (GPU) [Nvidia L4 24GB](https://www.nvidia.com/en-us/data-center/l4/)  | Float64   | 3m   |Right    |
| (GPU) [Nvidia L40S 48GB](https://www.nvidia.com/en-us/data-center/l40s/)  | Float64   | Cell C   |Right    |
| (GPU) [Nvidia A100 40GB](https://www.nvidia.com/en-us/data-center/a100/)  | Float64   | Cell C   |Right    |
| (GPU) [Nvidia A100 80GB](https://www.nvidia.com/en-us/data-center/a100/)  | Float64   | Cell C   |Right    |
| (GPU) [Nvidia H100 SXM 80GB](https://www.nvidia.com/en-us/data-center/h100/)  | Float64   | Cell C   |Right    |
| (GPU) [Nvidia H100 NVL 96GB](https://www.nvidia.com/en-us/data-center/h100/)  | Float64   | Cell C   |Right    |
| (GPU) [Nvidia H200 SXM 96GB](https://www.nvidia.com/en-us/data-center/h200/)  | Float64   | Cell C   |Right    |
| (GPU) [Nvidia GH200 80GB](https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/grace-hopper-superchip-datasheet-partner)  | Float64   | Cell C   |Right    |
| (GPU) [Nvidia B200 80GB](https://www.nvidia.com/en-us/data-center/dgx-b200/)  | Float64   | Cell C   |Right    |


