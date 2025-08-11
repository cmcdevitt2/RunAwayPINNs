'''
This script computes the time trajectory of the RPF, the residual of the PDE,
and the decay rate from a trained PINN model in TrainDecayPINN.py
'''
# Importing relevant libraries
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
from matplotlib.colors import TwoSlopeNorm
from scipy import interpolate
from scipy.integrate import simpson as simps
from scipy.integrate import trapz
from scipy import integrate
from scipy.special import erf
import time
import copy
from TrainDecayPINN import *

plt.rcParams.update({'font.size': 18})

# Provide path to where the model and training/test data is saved
save_path_root = "./"

# Specify which model to load
ModelToLoad = 15000

# Provide path to model, training distribution and loss history
ckpt_save_path  = str(save_path_root) + f"./model.ckpt-{ModelToLoad}.pt"
train_save_path = str(save_path_root) + "train.dat"
test_save_path  = str(save_path_root) + "test.dat"
loss_save_path  = str(save_path_root) + "loss.dat"

# Specify Coulog logarithm to compute avalanche growth rate
CouLog = 15.0

# Define neural network values to plot the RPF at
EFVal    = -1.5   # electric field normalized to Ec
ZeffVal  =  2.0   # effective charge
alphaVal =  0.1  # synchrotron radiation tau_c/tau_s
tVal     = np.array([0, 0.5*tMax, 1*tMax]) # time normalized to tau_c


# Normalize training parameters to be between 0 and 1
EFValNorm    = ( EFVal    - EFMin    ) / ( EFMax    - EFMin    )
ZeffValNorm  = ( ZeffVal  - ZeffMin  ) / ( ZeffMax  - ZeffMin  )
alphaValNorm = ( alphaVal - alphaMin ) / ( alphaMax - alphaMin )
tValNorm     = ( tVal     - tMin     ) / ( tMax     - tMin     )

# load training points and loss history
trainpts = np.loadtxt(train_save_path)
testpts  = np.loadtxt(test_save_path )
loss     = np.loadtxt(loss_save_path )

# Load training points in (p,\xi) space
xtrainpts = trainpts[:,0]
ytrainpts = trainpts[:,1]


# Load loss history
steps   = loss[:,0]
lossPDE = loss[:,1]
lossBC  = loss[:,2]
testPDE = loss[:,3]
testBC  = loss[:,4]

def Computefe(pgrid,xigrid):
    numxi = len(xigrid)
    nump = len(pgrid)
    
    fe = np.zeros([numxi,nump])
    for i in range(0,numxi):
        fe[i,:] = np.exp( -(pgrid-p0Init)**2/DpInit**2 - (xigrid[i]-xi0Init)**2/DxiInit**2 ) / Ixi / Ip / 2 / np.pi

    return fe

# Evaluate number of electrons at a given time
def NumbernRE(model,EFVal,ZeffVal,alphaVal,tVal,fe):
    EFValNorm = ( EFVal - EFMin ) / ( EFMax - EFMin )
    ZeffValNorm = ( ZeffVal - ZeffMin ) / ( ZeffMax - ZeffMin )
    alphaValNorm = ( alphaVal - alphaMin ) / ( alphaMax - alphaMin )
    tValNorm = ( tVal - tMin ) / ( tMax - tMin )

    pgridNormInt = ( pgridInt - pMin ) / ( pMax - pMin )

    pnewNormInt, xinewInt = np.meshgrid(pgridNormInt,xigridInt)
    X = np.vstack((np.ravel(pnewNormInt), np.ravel(xinewInt))).T
    EFValVec = EFValNorm * np.ones([numpInt*numxiInt,1])
    ZeffValVec = ZeffValNorm * np.ones([numpInt*numxiInt,1])
    alphaValVec = alphaValNorm * np.ones([numpInt*numxiInt,1])
    tValVec = tValNorm * np.ones([numpInt*numxiInt,1])
    X = np.hstack((X, EFValVec))
    X = np.hstack((X, ZeffValVec))
    X = np.hstack((X, alphaValVec))
    X = np.hstack((X, tValVec))

    y_2 = model.predict(X)
    P_2 = y_2[:,0]
    P_2 = P_2.reshape(pnewNormInt.shape)
    
    integrand = 2 * np.pi * pgridInt**2 * fe * P_2

    nRE = simps(simps(integrand,x=pgridInt),x=xigridInt)
    
    return nRE

# Fit growth/decay rate to n_RE(t)
def EvalgavAndStdDev(nREfit,tgridfit):
    numt = len(tgridfit)
    dtfit = tgridfit[1] - tgridfit[0]

    dnRE_t = np.zeros(numtfit)
    gavEst = np.zeros(numtfit)

    dnRE_t[0] = (nREfit[1] - nREfit[0]) / dtfit
    gavEst[0] = dnRE_t[0] / nREfit[0]
    for i in range(1,numtfit-1):
        dnRE_t[i] = (nREfit[i+1] - nREfit[i-1]) / (2*dtfit)
        gavEst[i] = dnRE_t[i] / nREfit[i]

    dnRE_t[-1] = (nREfit[numtfit-1] - nREfit[numtfit-2]) / dtfit
    gavEst[-1] = dnRE_t[-1] / nREfit[-1]

    # Compute average
    gavEstavg = 0
    for i in range(0,numtfit):
        gavEstavg += gavEst[i] / numtfit

    # Compute variance
    vargavEst = 0
    for i in range(0,numtfit):
        vargavEst += (gavEstavg-gavEst[i])**2 / numtfit

    StdDevgavEstEF = np.sqrt(vargavEst)

    return gavEstavg, StdDevgavEstEF


data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [],
)

model = dde.Model(data, net)
model.compile("L-BFGS-B")
model.restore(save_path = ckpt_save_path, verbose=1)





# number of grid points used for performing integrals
numpInt = 100
numxiInt = 80
pgridInt = np.linspace(pMin,pMax,numpInt)
xigridInt = np.linspace(xiMin,xiMax,numxiInt)
ggridInt = np.sqrt(pgridInt**2+1)

# parameters used to define the initial electron distribution
DpInit  = 0.1 * pMax
DxiInit = 0.1
p0Init  = 0.75 * pMax
xi0Init = 0

# Define initial electron distribution
Ixi = np.sqrt(np.pi) / 2 * DxiInit * ( erf((xi0Init+1)/DxiInit) - erf((xi0Init-1)/DxiInit)  )
Ip = np.sqrt(np.pi) / 4 * (2*p0Init**2+DpInit**2) * (DpInit*erf(p0Init/DpInit) + DpInit) + p0Init*DpInit**2/2 * np.exp(-p0Init**2/DpInit**2)


'''
Compute and plot results
'''
fig, ax = plt.subplots(ncols=3,nrows=3, figsize=(18,12))
fig.set_tight_layout(True)

###################################################
# Predict RPF as a function p and \xi at given time
###################################################

nump = 100
numxi = 50

pgrid = np.linspace(pMin,pMax,nump)
xigrid = np.linspace(xiMin,xiMax,numxi)

pgridNorm = (pgrid - pMin) / (pMax-pMin)

ggrid = np.sqrt(1+pgrid**2)
for i in range(len(tVal)):
    
    pnewNorm, xinew = np.meshgrid(pgridNorm,xigrid)
    X = np.vstack((np.ravel(pnewNorm), np.ravel(xinew))).T
    EFValVec = EFValNorm * np.ones([nump*numxi,1])
    ZeffValVec = ZeffValNorm * np.ones([nump*numxi,1])
    alphaValVec = alphaValNorm * np.ones([nump*numxi,1])
    tValVec = tValNorm[i] * np.ones([nump*numxi,1])
    X = np.hstack((X, EFValVec))
    X = np.hstack((X, ZeffValVec))
    X = np.hstack((X, alphaValVec))
    X = np.hstack((X, tValVec))
    
    # Predict RPF
    t1 = time.time()
    Pnew = model.predict(X)[:,0].reshape(pnewNorm.shape)
    t2 = time.time()
    print(f'Time per prediction of RPF: {(t2-t1)/len(pnewNorm.flatten()):.4e} seconds')
    
    # Compute residual
    resnew = model.predict(X,operator=pde)[:,0].reshape(pnewNorm.shape)

    cs1 = ax[0,i].contourf(1e-6*mecSQ*(ggrid-1), xigrid, Pnew, levels=50, cmap='jet')
    ax[0,i].contour(1.e-6*mecSQ*(ggrid-1), xigrid, Pnew, [0.5], colors='black')
    fig.colorbar(cs1,ax=ax[0,i])
    ax[0,i].set_ylabel("$\\xi$")
    ax[0,i].set_xlabel("Energy [MeV]")
    ax[0,i].set_title(f"RPF$(t={tVal[i]:.1f})$")
    ax[0,i].set_xticks([1,2,3,4,5])

    cs2 = ax[1,i].contourf(1.e-6*mecSQ*(ggrid-1), xigrid, resnew, 50,cmap='jet')
    
    fig.colorbar(cs2,ax=ax[1,i])
    ax[1,i].set_ylabel("$\\xi$")
    ax[1,i].set_xlabel("Energy [MeV]")
    ax[1,i].set_title(f"residual of RPF$(t={tVal[i]:.1f})$")
    ax[1,i].set_xticks([1,2,3,4,5])


# Computes evolution of primary n_RE

numt = 20
tgrid = np.linspace(tMin,tMax,numt)
nREvst = np.zeros(numt)
fe = Computefe(pgridInt,xigridInt)
t1 = time.time()
for i in range(0,numt):
    nREvst[i] = NumbernRE(model,EFVal,ZeffVal,alphaVal,tgrid[i],fe)

nREfit, tgridfit = nREvst[int(numt/2):], tgrid[int(numt/2):]
numtfit = len(tgridfit)
gavEstavg, StdDevgavEstEF = EvalgavAndStdDev(nREfit,tgridfit)
t2 = time.time()

print('')
print(f'Time to predict decay rate: {(t2-t1):.4e} seconds')
print(f'E/Ec = {abs(EFVal)}, Zeff = {ZeffVal}, alpha = {alphaVal}')
print(f'Decay growth rate normalized to tauc: {gavEstavg}')

ax[2,0].plot(tgrid, nREvst/nREvst[0],linestyle='-',color='blue',linewidth=2)
ax[2,0].plot(tgridfit, nREfit[0]*np.exp(gavEstavg*tgridfit)/ np.exp(gavEstavg*tgridfit[0]),
            linewidth=2, color='red', linestyle='dotted')
ax[2,0].set_ylabel("$n_{RE}(t)$")
ax[2,0].set_xlabel("$t/\\tau_c$")
ax[2,0].set_yscale("log")
ax[2,0].set_xticks([0,1,2,3,4,5])


ptrainpts = pMin + (pMax-pMin) * xtrainpts
gtrainpts = np.sqrt(ptrainpts**2+1)
Energytrainpts = mecSQ * (gtrainpts-1)

ax[2,1].scatter(1.e-6*Energytrainpts, ytrainpts,s=0.0001,color='black')
ax[2,1].set_ylabel("$\\xi$")
ax[2,1].set_xlabel("Energy [MeV]")
ax[2,1].set_title("Training Points")
ax[2,1].set_xticks([1,2,3,4,5])


# Removes redundant values in test training array
testPDEClean = np.copy(testPDE)
for i in range(1,len(lossPDE)):
    if np.isclose(lossPDE[i], testPDE[i]):
        testPDEClean[i] = 'nan'

testBCClean = np.copy(testBC)
for i in range(1,len(lossBC)):
    if np.isclose(lossBC[i], testBC[i]):
        testBCClean[i] = 'nan'

ax[2,2].plot(1e-3*steps, lossPDE, label='training PDE', linestyle='-',color='blue',linewidth=2)
ax[2,2].plot(1e-3*steps, lossBC, label='training BC', linestyle='-',color='red',linewidth=2)
ax[2,2].plot(1e-3*steps, testPDEClean, 'xb', label='test PDE')
ax[2,2].plot(1e-3*steps, testBCClean, 'xr', label='test BC')

ax[2,2].set_xlabel("Thousands of epochs")
ax[2,2].set_title("Loss History")
ax[2,2].set_yscale("log")
ax[2,2].set_ylim(1e-9, 1e3)
ax[2,2].legend(loc='upper right',fontsize=14)

fig.savefig('DecayPINN_Results')
