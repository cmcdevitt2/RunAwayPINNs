This directory contains a physics informed neural network (PINN) that predicts the parametric dependence of the exponential "avalanche" growth rate of relativistic electrons (runaway electrons, RE) $\gamma_{av}$, on the plasma's parallel electric field strength $E_\Vert$, effective charge $Z_{eff}$, and synchrotron radiation strength $\alpha$. Further details on the formulation of the PINN is found in the [paper](https://doi.org/10.1017/S0022377824000679). 

We note that due to the decreasing support of Tensorflow ([Nvidia no longer will provide pre-built containers](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-02.html)), we have gone ahead and converted the backend of the PINN script from Tensorflow to PyTorch, which does not impact the overall performance, as simple tensor operations are done for this script, and the L-BFGS-B optimizer is from Scipy, which is independent of the backend.

To launch the script that trains the PINN, simply run the command ```DDEBACKEND=pytorch python TrainAvalanchePINN.py```, assuming the environment created in the parent directory of this repo is activated. Assuming the random seed and other parameters were not change, the output should look like what is shown below:
````
Using backend: pytorch
Other supported backends: tensorflow.compat.v1, tensorflow, jax, paddle.
paddle supports more examples now and is recommended.
Set the default float type to float64
Warning: 1000000 points required, but 1194102 points sampled.
Compiling model...
'compile' took 0.980408 s

Training model...

Step      Train loss              Test loss               Test metric
0         [7.84e-02, 2.00e-01]    [7.31e-02, 2.00e-01]    []  
1000      [1.14e-02, 8.26e-04]    [5.66e-03, 8.26e-04]    []  
2000      [7.86e-03, 4.90e-04]    [3.47e-03, 4.90e-04]    []  
3000      [5.57e-03, 3.45e-04]    [2.59e-03, 3.45e-04]    []  
4000      [3.64e-03, 2.16e-04]    [1.76e-03, 2.16e-04]    []  
5000      [2.41e-03, 8.31e-05]    [1.12e-03, 8.31e-05]    []  
6000      [1.74e-03, 4.68e-05]    [8.38e-04, 4.68e-05]    []  
7000      [1.32e-03, 4.78e-05]    [7.29e-04, 4.78e-05]    []  
8000      [8.15e-04, 2.80e-05]    [3.61e-04, 2.80e-05]    []  
9000      [6.08e-04, 2.09e-05]    [2.66e-04, 2.09e-05]    []  
10000     [5.55e-04, 2.80e-05]    [3.12e-04, 2.80e-05]    []  
11000     [4.35e-04, 1.76e-05]    [2.49e-04, 1.76e-05]    []  
12000     [2.68e-04, 2.66e-05]    [1.25e-04, 2.66e-05]    []  
13000     [2.60e-04, 1.33e-05]    [1.40e-04, 1.33e-05]    []  
14000     [3.35e-04, 2.34e-05]    [1.36e-04, 2.34e-05]    []  
15000     [2.05e-04, 1.05e-05]    [1.07e-04, 1.05e-05]    []  

Best model at step 15000:
  train loss: 2.15e-04
  test loss: 1.18e-04
  test metric: []

Epoch 15000: saving model to ./model.ckpt-15000.pt ...

'train' took 1602.705070 s

Saving loss history to /home/users/u0001781/RunAwayPINNs/Surrogates/AvalanchePINN/loss.dat ...
Saving training data to /home/users/u0001781/RunAwayPINNs/Surrogates/AvalanchePINN/train.dat ...
Saving test data to /home/users/u0001781/RunAwayPINNs/Surrogates/AvalanchePINN/test.dat ...
Compiling model...
'compile' took 0.000196 s

Training model...

Step      Train loss              Test loss               Test metric
15000     [2.05e-04, 1.05e-05]    [1.07e-04, 1.05e-05]    []  

````


Once the PINN has trained. Run the following command ```DDEBACKEND=pytorch python ComputeAvalanchePINN_Results.py``` to compute the RPF, residual, and avalanche growth rate for a chosen set of paramters $E_\Vert$,$Z_{eff}$,$\alpha$. The time per prediction in computing the RPF and the time taken to evaluate the avalanche growth rate is also computed. An example output of the plotting script is shown below:

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
![RPF](AvalanchePINN_Results.png)
