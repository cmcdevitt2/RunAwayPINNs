'''
This script computes the RPF, Residual, and avalanche growth rate
from a trained PINN model in TrainAvalanchePINN.py
'''

# Importing relevant libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simps
from TrainAvalanchePINN import *
import time

plt.rcParams.update({'font.size': 18})

# Provide path to where the model and training/test data is saved
save_path_root = "./"

# Specify which model to load
ModelToLoad = 188510

# Provide path to model, training distribution and loss history
ckpt_save_path  = str(save_path_root) + f"./model.ckpt-{ModelToLoad}.pt"
train_save_path = str(save_path_root) + "train.dat"
test_save_path  = str(save_path_root) + "test.dat"
loss_save_path  = str(save_path_root) + "loss.dat"

# Physical Constants
clight      = 2.99792e10 # speed of lightin cm/s
CLASSICALER = 2.8179e-13 # classical electron radius in units of cm

# Define neural network to plot the RPF at
EFValPlot    = -3.0 # electric field normalized to Ec
ZeffValPlot  =   1.0 # effective charge
alphaValPlot =   0.2 # synchrotron radiation tau_c/tau_s

# Specify Coulog logarithm to compute avalanche growth rate
CouLog = 15.0


# load training points and loss history
trainpts = np.loadtxt(train_save_path)
testpts  = np.loadtxt(test_save_path )
loss     = np.loadtxt(loss_save_path )

# Load training points in (p,\xi) space
xtrainpts = trainpts[:,0]
ytrainpts = trainpts[:,1]

# Load test points in (p,\xi) space
xtestpts = testpts[:,0]
ytestpts = testpts[:,1]

# Load loss history
steps   = loss[:,0]
lossPDE = loss[:,1]
lossBC  = loss[:,2]
testPDE = loss[:,3]
testBC  = loss[:,4]

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

# Function to compute avalanche growth rate
def GetAvalanche(EFVal, ZeffVal, alphaVal):
    # Normalize training parameters to be between 0 and 1
    EFValNorm    = ( EFVal - EFMin ) / ( EFMax - EFMin )
    ZeffValNorm  = ( ZeffVal - ZeffMin ) / ( ZeffMax - ZeffMin )
    alphaValNorm = ( alphaVal - alphaMin ) / ( alphaMax - alphaMin )

    nump  = 1000  # number of momentum points
    pgrid = np.logspace(np.log10(pMin),np.log10(pMax),nump)
    ggrid = np.sqrt(1+pgrid**2)
    vgrid = pgrid / ggrid
    xi1 = - np.sqrt((ggrid-1)/(ggrid+1)) # Rosenbluth-Putvinski pitch-angle source

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
    
    Pnew = model.predict(X)[:,0].reshape(pgrid.shape)
    
    # Evaluate integrand of avalanche growth rate
    integrand = 1/2/CouLog*pgrid**2 * vgrid / (ggrid**2-1) / (ggrid-1)**2 * Pnew
    
    # Carry out integration
    GammaRPsimps = simps(integrand,x=pgrid)

    return GammaRPsimps

# Compute single value of avalanche growth rate
t1 = time.time()
gAv = GetAvalanche(EFValPlot, ZeffValPlot, alphaValPlot)
t2 = time.time()
print(f'E/Ec = {abs(EFValPlot)}, Zeff = {ZeffValPlot}, alpha = {alphaValPlot}')
print(f'Avalanche growth rate normalized to tauc: {gAv}, time to predict: {(t2-t1)*1e3:.2f} ms')

# Plot RPF at specific values defined in beginning of script
nump  = 200  # number of momentum points
numxi = 150  # number of pitch-angle points

# Normalize training parameters to be between 0 and 1
EFValNorm    = ( EFValPlot - EFMin ) / ( EFMax - EFMin )
ZeffValNorm  = ( ZeffValPlot - ZeffMin ) / ( ZeffMax - ZeffMin )
alphaValNorm = ( alphaValPlot - alphaMin ) / ( alphaMax - alphaMin )


# Create grids
pgrid  = np.logspace(np.log10(pMin),np.log10(pMax),nump)
xigrid = np.linspace(xiMin,xiMax,numxi)
ggrid  = np.sqrt(1+pgrid**2)

# Mesh grid in 2D
pnew, xinew = np.meshgrid(pgrid,xigrid)

# Create array of the specific parameters
EFValVec    = EFValNorm * np.ones([nump*numxi,1])
ZeffValVec  = ZeffValNorm * np.ones([nump*numxi,1])
alphaValVec = alphaValNorm * np.ones([nump*numxi,1])

# Create 5D array that the RPF will predict for
X = np.vstack((np.ravel(pnew), np.ravel(xinew))).T
X = np.hstack((X, EFValVec   ))
X = np.hstack((X, ZeffValVec ))
X = np.hstack((X, alphaValVec))

# Predict solution
t1   = time.time()
Pnew = model.predict(X)[:,0].reshape(pnew.shape)
t2   = time.time()
print(f'Time per prediction of RPF: {(t2-t1)/len(pnew.flatten()):.4e} seconds')

# Evaluate residual of the PDE
resnew = model.predict(X,operator=pde)[:,0].reshape(pnew.shape)



# Scan electric field at given value of Zeff and alpha
numEF = 50
EFScan = np.linspace(EFMin, EFMax, numEF)
gAvScanEF = np.zeros(numEF)
ZeffValScan, alphaValScan = 1, 0.1
for i in range(numEF):
    gAvScanEF[i] = GetAvalanche(EFScan[i], ZeffValScan, alphaValScan)

gAvRP = 1/CouLog*np.sqrt(np.pi/(3*(ZeffValScan+5)))*(abs(EFScan)-1)

Fig7bMcDevitt_2018_EF  = [1.75, 2, 2.25, 2.5, 2.75, 3]
Fig7bMcDevitt_2018_gav = [0.007165530345367155, 0.022686426125567732, 0.032885477102533096, 0.04200493817125091, 0.05073909995224403, 0.059087454403202705]

Fig4McDevitt_2019_EF  = [1.6831807977147843, 3.671641963654552, 5.669782913215883, 7.658972665019637, 9.666157331178418]
Fig4McDevitt_2019_gav = [ 0.00030487848951421537, 0.0845480526952617, 0.1464544323029529, 0.20643569723787894, 0.267186314150612]

EFBenchmark  = np.concatenate((Fig7bMcDevitt_2018_EF,Fig4McDevitt_2019_EF),axis=0)
gavBenchmark = np.concatenate((Fig7bMcDevitt_2018_gav,Fig4McDevitt_2019_gav),axis=0)


''' 
Plot Results
'''
fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(10,8))
fig.set_tight_layout(True)

# Plot scan of avalanche growth rate
ax[0,0].plot(abs(EFScan), gAvScanEF, lw=2, c='black', label='PINN')
ax[0,0].plot(abs(EFScan), gAvRP, lw=2, c='green', ls='dashed', label='R-P')
ax[0,0].plot(abs(EFBenchmark), gavBenchmark, 'o', mfc='none',mew=2, c='blue', label='RAMc')
ax[0,0].set_xlabel('$E/E_c$')
ax[0,0].set_ylabel('$\\gamma_{av}$')
ax[0,0].legend(fontsize=14)

# Plot Loss history
ax[0,1].plot(steps/1e3, lossPDE, label='PDE training loss', linestyle='-' ,color='blue',linewidth=2)
ax[0,1].plot(steps/1e3, lossBC , label='BC training loss' , linestyle='-' ,color='red' ,linewidth=2)
ax[0,1].plot(steps/1e3, testPDE, label='PDE test loss'    , linestyle='--',color='blue',linewidth=2)
ax[0,1].plot(steps/1e3, testBC , label='BC test loss'     , linestyle='--',color='red' ,linewidth=2)
ax[0,1].set_xlabel("thousands of steps")
ax[0,1].set_title("Loss History")
ax[0,1].set_yscale("log")
ax[0,1].legend(fontsize=14)

# RPF at specified parameters
cs0 = ax[1,0].contourf(mecSQ*(ggrid-1), xigrid, Pnew, levels=50, cmap='jet')
fig.colorbar(cs0,ax=ax[1,0],shrink=0.625)
ax[1,0].set_ylabel("$\\xi$")
ax[1,0].set_xlabel("Energy [eV]")
ax[1,0].set_xscale('log')
ax[1,0].axis('scaled')
ax[1,0].set_yticks([-1,-0.5,0,0.5,1])
ax[1,0].set_title(f"RPF $(E_\\Vert={abs(EFValPlot)}, Z_{{eff}}={ZeffValPlot}, \\alpha={alphaValPlot})$",fontsize=12)

# Plot Residual
cs1 = ax[1,1].contourf(mecSQ*(ggrid-1), xigrid, resnew, 50,cmap='jet')
fig.colorbar(cs1,ax=ax[1,1],shrink=0.625)
ax[1,1].set_xscale("log")
ax[1,1].set_ylabel("$\\xi$")
ax[1,1].set_xlabel("Energy [eV]")
ax[1,1].axis('scaled')
ax[1,1].set_title("residual of RPF", fontsize=14)
ax[1,1].set_yticks([-1,-0.5,0,0.5,1])

# Save figure
fig.savefig('AvalanchePINN_Results')
