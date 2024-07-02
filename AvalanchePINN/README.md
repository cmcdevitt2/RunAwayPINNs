** UNDER CONSTRUCTION **

This directory contains a physics informed neural network (PINN) that predicts the parametric dependence of the exponential "avalanche" growth rate of relativistic electrons (runaway electrons, RE) $\gamma_{av}$, on the plasma's parallel electric field strength $E_\Vert$, effective charge $Z_{eff}$, and synchrotron radiation strength $\alpha$. Further details on the formulation of the PINN is found in the [manuscript](https://doi.org/10.48550/arXiv.2403.04948). To launch the script that trains the PINN, simply run the **RPF.py** python script, assuming the environment created in the parent directory of this repo is activated. If the correct libraries and modules are not loaded, the script will typically either throw an error or run inefficiently (i.e the GPU will not be utilized). 

The PINN training script will load in the following options from the **config.py** file, which contains the following parameters:

- Flag to use (1) or not use (0) the default parameters
- Number of training epochs for the ADAM optimizer
- Number of desired training periods and training epochs per training period for the LBFGS-B optimizer
- Range of training inputs ($p/m_ec, \xi, E_\Vert/E_c, Z_{eff}, \alpha$)
- Neural network architecture (activation function, number of hidden layers, and number of neurons per hidden layer)

The following parameters are chosen as the default values

- 30,000 epochs for the ADAM optimizer
- 100 training periods of 10,000 training epochs for the LBFGS-B optimizer
- 4 hidden layers and 64 neurons per layer, and the $\tanh$ activation function
- Normalized momentum range corresponding to $p/m_ec$ from $0.25/\sqrt{E_\Vert/E_c-1}$ to 10
- $\xi$ from -1 to 1
- $E_\Vert/E_c$ from 1 to 500
- $Z_{eff}$ from 1 to 10
- $\alpha$ from 0 to 0.5
