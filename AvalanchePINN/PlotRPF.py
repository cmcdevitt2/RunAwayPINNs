'''
This script computes the RPF, Residual, and avalanche growth rate
from a trained PINN model in RPF_paper.py
'''

# Hiding unnecessary warnings
import warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# Importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import deepxde as dde
from deepxde.backend import tf
from scipy import integrate
import RPF_paper 
import time

plt.rcParams.update({'font.size': 18}) # update fontsize

# Load relevant params from training script
params = RPF_paper.Params()

pde = params.pde # pde function

EFMin = params.EFMin # minimum of electric field range
EFMax = params.EFMax # maxiumum of electric field range

ZeffMin = params.ZeffMin # minimum of effective charge range
ZeffMax = params.ZeffMax # maximum of effective charge range

alphaMin = params.alphaMin # minimum of synchrotron radiation strength range
alphaMax = params.alphaMax # maximum of synchrotron radiation strength range

EnergyMineV = params.EnergyMineV # minimum of energy range
EnergyMaxeV = params.EnergyMaxeV # maximum of energy range

xiMin = params.xiMin # minimum of pitch-angle range
xiMax = params.xiMax # minimum of pitch-angle range

# Lorentz factor
gMax = EnergyMaxeV/511e3 + 1
gMin = EnergyMineV/511e3 + 1

# momentum normalized to m_e*c
pMin = np.sqrt(gMin**2-1)
pMax = np.sqrt(gMax**2-1)

net = params.net # neural network
geom = params.geom # geometry of input domain
output_transform = params.output_transform # additional hidden layer to embedd constaints

# Provide path to data directory
save_path_root = "./"

# Specify which model to load
ModelToLoad = 255027

# Provide path to model, training distribution and loss history
ckpt_save_path = str(save_path_root) + f"model/model.ckpt-{ModelToLoad}.ckpt"
train_save_path = str(save_path_root) + "train.dat"
loss_save_path = str(save_path_root) + "loss.dat"

# Physical Constants
clight = 2.99792e10 # speed of lightin cm/s
CLASSICALER = 2.8179e-13 # classical electron radius in units of cm
mecSQ = 510.9989461e3 # electron rest mass in units eV
IA = 0.017045e6 # Alfven current in Amperes

# Define values to plot the RPF at
EFVal = -3 # electric field normalized to Ec
ZeffVal = 1 # effective charge
alphaVal = 0 # synchrotron radiation tau_c/tau_s

# Specify Coulog logarithm
CouLog = 15

# Normalize training parameters to be between 0 and 1
EFValNorm = ( EFVal - EFMin ) / ( EFMax - EFMin )
ZeffValNorm = ( ZeffVal - ZeffMin ) / ( ZeffMax - ZeffMin )
alphaValNorm = ( alphaVal - alphaMin ) / ( alphaMax - alphaMin )

# load training points and loss history
trainpts = np.loadtxt(train_save_path)
loss = np.loadtxt(loss_save_path)

# Load training points
xtrainpts = trainpts[:,0]
ytrainpts = trainpts[:,1]
ztrainpts = trainpts[:,2]

# Load loss history
steps = loss[:,0]
lossPDE = loss[:,1]
lossBC = loss[:,2]
testPDE = loss[:,3]
testBC = loss[:,4]

# Plot Training point distribution
fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
fig7.set_tight_layout(True)
ax7.scatter(xtrainpts, ytrainpts,s=0.01,color='black')
ax7.set_ylabel("$\\xi$")
ax7.set_xlabel("$p/m_ec$")
ax7.set_title("Training Points")
fig7.savefig('TrainingPoints')

# Plot Loss history
fig8, ax8 = plt.subplots(num=8,nrows=1,ncols=1, clear=True)
fig8.set_tight_layout(True)
ax8.plot(steps/1e3, lossPDE, label='PDE training loss', linestyle='-',color='blue',linewidth=2)
ax8.plot(steps/1e3, lossBC, label='BC training loss', linestyle='-',color='red',linewidth=2)
ax8.plot(steps/1e3, testPDE, label='PDE test loss', linestyle='--',color='blue',linewidth=2)
ax8.plot(steps/1e3, testBC, label='BC test loss', linestyle='--',color='red',linewidth=2)
ax8.set_xlabel("thousands of steps")
ax8.set_title("Loss History")
ax8.set_yscale("log")
ax8.legend()
fig8.savefig('LossHitory')


# Restoring model
losses = []
data = dde.data.TimePDE(
    geom,
    pde,
    losses)

loss_weights = [1]
loss = ["MSE"]

model = dde.Model(data, net)
model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)    
model.restore(save_path = ckpt_save_path, verbose=1)

# Plot RPF at specific values defined in beginning of script
nump = 200  # number of momentum points
numxi = 150 # number of pitch-angle points

# Create grids
pgrid = np.logspace(np.log10(pMin),np.log10(pMax),nump)
xigrid = np.linspace(xiMin,xiMax,numxi)
ggrid = np.sqrt(1+pgrid**2)

# Mesh grid in 2D
pnew, xinew = np.meshgrid(pgrid,xigrid)

# Create array of the specific parameters
EFValVec = EFValNorm * np.ones([nump*numxi,1])
ZeffValVec = ZeffValNorm * np.ones([nump*numxi,1])
alphaValVec = alphaValNorm * np.ones([nump*numxi,1])

# Create 5D array that the RPF will predict for
X = np.vstack((np.ravel(pnew), np.ravel(xinew))).T
X = np.hstack((X, EFValVec))
X = np.hstack((X, ZeffValVec))
X = np.hstack((X, alphaValVec))

# Predict solution
t1 = time.time()
y_pred = model.predict(X)
t2 = time.time()
Pnew = y_pred[:,0]
Pnew = Pnew.reshape(pnew.shape)
print(f'Time per prediction of RPF: {(t2-t1)/len(pnew.flatten()):.4e} seconds')

# RPF at specified parameters
fig1, ax1 = plt.subplots(num=1,nrows=1,ncols=1, clear=True)
fig1.set_tight_layout(True)
cs1 = ax1.contourf(mecSQ*(ggrid-1), xigrid, Pnew, levels=30, cmap='jet')
fig1.colorbar(cs1,ax=ax1)
ax1.set_ylabel("$\\xi$")
ax1.set_xlabel("Energy [eV]")
ax1.set_xscale('log')
ax1.set_title("RPF")
fig1.savefig('RPF')


# Evaluate residual of the PDE
res_pred = model.predict(X,operator=pde)
resnew = res_pred[:,0]
resnew = resnew.reshape(pnew.shape)

# Plot Residual
fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)
cs2 = ax2.contourf(mecSQ*(ggrid-1), xigrid, resnew, 50,cmap='jet')
fig2.colorbar(cs2,ax=ax2)
ax2.set_xscale("log")
ax2.set_ylabel("$\\xi$")
ax2.set_xlabel("Energy [eV]")
ax2.set_title("residual of RPF")
fig2.savefig('Residual')

# Compute avalanche growth rate

nump = 1000  # number of momentum points
pgrid = np.logspace(np.log10(pMin),np.log10(pMax),nump)
ggrid = np.sqrt(1+pgrid**2)
vgrid = pgrid / ggrid
xi1 = - np.sqrt((ggrid-1)/(ggrid+1))# Rosenbluth-Putvinski pitch-angle source

# Create grids of training paramters to predict
EFValVec = EFValNorm    * np.ones([nump,1])
ZeffVec  = ZeffValNorm  * np.ones([nump,1])
alphaVec = alphaValNorm * np.ones([nump,1])

# Create array for PINN to predict
X = np.vstack((pgrid, xi1 )).T
X = np.hstack((X, EFValVec))
X = np.hstack((X, ZeffVec ))
X = np.hstack((X, alphaVec))

# Make prediction
t1 = time.time()
y_pred = model.predict(X)
t2 = time.time()
print(f'Time to predict avalanche growth rate: {(t2-t1):.4e} seconds')

Pnew = y_pred[:,0]
Pnew = Pnew.reshape(pgrid.shape)

# Evaluate integrand of avalanche growth rate
integrand = 1/2/CouLog*pgrid**2 * vgrid / (ggrid**2-1) / (ggrid-1)**2 * Pnew

# Carry out integration
GammaRPsimps = simps(integrand,pgrid)
print(f'E/Ec = {abs(EFVal)}, Zeff = {ZeffVal}, alpha = {alphaVal}')
print(f'Avalanche growth rate normalized to tauc: {GammaRPsimps}')
