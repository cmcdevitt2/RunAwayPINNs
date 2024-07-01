** UNDER CONSTRUCTION !!! **

# RunAwayPINNs
This repository contains Physics Informed Neural Networks (PINNs) [[Raissi 2019](https://doi.org/10.1016/j.jcp.2018.10.045) , [Cuomo 2022](https://doi.org/10.1007/s10915-022-01939-z)] that describe the formation of relativistic electrons (Runaway Electrons, RE). Specifically, two PINNs are presently available, describing a primary generation mechanism known as "hot-tail formation" [[McDevitt 2023](https://doi.org/10.1063/5.0164712)], and secondary generation known as the "runaway electron avalanche" [[Arnaud 2024](https://doi.org/10.48550/arXiv.2403.04948)]. Both PINNs solve an adjoint of the Fokker-Planck equation that describes the probability of an electron running away $P$ as a function of energy, parallel direction of momentum $\xi \equiv p_\Vert/p$, and  other relevant physics parameters. Below is an example of the PINN learning the probability solution. 

![RPF-gif](RPF-animation.gif)

## Getting started
The main libraries used to construct and train the PINN include [DeepXDE](https://deepxde.readthedocs.io/en/latest/) with the [Tensorflow](https://www.tensorflow.org) backend. While the PINN can be trained on either a CPU or a GPU, it is recommended to use GPUs for training, given their increased performance for neural network training. The PINNs in this repo can be trained on a single GPU. To get started the bash script [init.sh](init.sh) will create a Python environment and install the relevant dependencies. 
