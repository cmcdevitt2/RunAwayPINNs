# AvalanchePINN
This directory contains a physics-informed neural network (PINN) that predicts the parametric dependence of the exponential "avalanche" growth rate of relativistic electrons (runaway electrons, RE) $\gamma_{av}$, on the plasma's parallel electric field strength $E_\Vert$, effective charge $Z_{eff}$, and synchrotron radiation strength $\alpha$. Further details on the formulation of the PINN is found in the [paper](https://doi.org/10.1017/S0022377824000679). 

We note that due to the decreasing support of Tensorflow ([Nvidia no longer will provide pre-built containers](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-02.html)), we have gone ahead and converted the backend of the PINN script from Tensorflow to PyTorch, which does not impact the overall performance, as simple tensor operations are done for this script, and the L-BFGS-B optimizer is from Scipy, which is independent of the backend.

# Getting started and example solution
To launch the script that trains the PINN, simply run the command ```DDEBACKEND=pytorch python TrainAvalanchePINN.py```, assuming the environment created in the parent directory of this repo is activated. Assuming the existing script is ran exactly as is, an example solution is shown below, where this example trained till [REPLACE_WHEN_DONE] iterations. Here, the first 15,000 iterations are with the ADAM optimizer, and the remaining iterations are with the L-BFGS-B optimizer. We note that the example script uses approximately 10 GB of GPU memory, so please use a GPU that has sufficient memory to run the script as is. Below is an output of the training script:
````
Using backend: pytorch
Other supported backends: tensorflow.compat.v1, tensorflow, jax, paddle.
paddle supports more examples now and is recommended.
Set the default float type to float64
Warning: 1000000 points required, but 1194102 points sampled.
Compiling model...
'compile' took 1.798237 s

Training model...

Step      Train loss              Test loss               Test metric
0         [6.48e-02, 3.39e-01]    [6.11e-02, 3.39e-01]    []  
1000      [1.90e-02, 1.76e-03]    [1.25e-02, 1.76e-03]    []  
2000      [1.19e-02, 8.98e-04]    [6.20e-03, 8.98e-04]    []  
3000      [9.49e-03, 6.74e-04]    [4.31e-03, 6.74e-04]    []  
4000      [7.83e-03, 4.99e-04]    [3.58e-03, 4.99e-04]    []  
5000      [6.43e-03, 4.69e-04]    [3.18e-03, 4.69e-04]    []  
6000      [5.07e-03, 3.36e-04]    [2.64e-03, 3.36e-04]    []  
7000      [3.77e-03, 2.05e-04]    [1.92e-03, 2.05e-04]    []  
8000      [2.74e-03, 1.32e-04]    [1.35e-03, 1.32e-04]    []  
9000      [2.11e-03, 7.74e-05]    [1.05e-03, 7.74e-05]    []  
10000     [1.62e-03, 5.34e-05]    [8.15e-04, 5.34e-05]    []  
11000     [1.29e-03, 3.92e-05]    [6.57e-04, 3.92e-05]    []  
12000     [1.04e-03, 4.49e-05]    [5.54e-04, 4.49e-05]    []  
13000     [8.89e-04, 2.67e-05]    [4.61e-04, 2.67e-05]    []  
14000     [7.29e-04, 3.01e-05]    [3.79e-04, 3.01e-05]    []  
15000     [6.27e-04, 2.54e-05]    [3.22e-04, 2.54e-05]    []  

Best model at step 15000:
  train loss: 6.53e-04
  test loss: 3.48e-04
  test metric: []

Epoch 15000: saving model to ./model.ckpt-15000.pt ...

'train' took 607.897830 s

Saving loss history to /home/users/u0001781/RunAwayPINNs/Surrogates/AvalanchePINN/loss.dat ...
Saving training data to /home/users/u0001781/RunAwayPINNs/Surrogates/AvalanchePINN/train.dat ...
Saving test data to /home/users/u0001781/RunAwayPINNs/Surrogates/AvalanchePINN/test.dat ...
Compiling model...
'compile' took 0.000154 s

Training model...

Step      Train loss              Test loss               Test metric
15000     [6.27e-04, 2.54e-05]    [3.22e-04, 2.54e-05]    []  

````


Once the PINN has trained. Run the following command ```DDEBACKEND=pytorch python PredictAvalanchePINN.py``` to compute the RPF, residual, and avalanche growth rate for a chosen set of paramters $(E_\Vert,Z_{eff},\alpha)$. The time per prediction in computing the RPF and the time taken to evaluate the avalanche growth rate is also computed. An example output of the plotting script is shown below:

```
Using backend: pytorch
Other supported backends: tensorflow.compat.v1, tensorflow, jax, paddle.
paddle supports more examples now and is recommended.
Set the default float type to float64
Compiling model...
'compile' took 0.000120 s

Restoring model from ././model.ckpt-15000.pt ...

Time to predict avalanche growth rate: 3.2780e-03 seconds
E/Ec = 3.0, Zeff = 1.0, alpha = 0.0
Avalanche growth rate normalized to tauc: 0.06587677105477978
Time per prediction of RPF: 2.4440e-06 seconds
```
Example outputs saved from the plotting script are shown below:
![AvalanchePINN_Results](AvalanchePINN_Results.png)
