import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
from matplotlib.colors import TwoSlopeNorm
from scipy import interpolate
from scipy.integrate import simps
from scipy.integrate import trapz
import deepxde as dde
from deepxde.backend import tf
from scipy import integrate
from scipy.special import kn
from scipy.special import erf
import time
import copy


dde.config.set_default_float("float64")


# Set path to folder with data
save_path_root = "/Volumes/Storage/ML_models/FokkerPlanck/REdecay/old_models/Test_Mar26_2025/"

# Select model to load
ckpt_save_path = str(save_path_root) + "model/model.ckpt-45003.ckpt"
train_save_path = str(save_path_root) + "data/train.dat"
loss_save_path = str(save_path_root) + "data/loss.dat"

mecSQ = 511e3 # electron rest mass in units eV

CouLog0 = 15

EnergyMaxeV = 5e6
EnergyMineV = 1e4

gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)
xiMax = 1
xiMin = -1
tMax = 5 # time in units of tau_c
tMin = 0

pRE = 0.25 * pMax

# Define range of parameters scanned
EFMax = -1 # normalized to E_c
EFMin = -3

ZeffMin = 1
ZeffMax = 2

alphaMin = 0
alphaMax = 0.1

Dp = 0.1*pMax # sets width of transition region in initial RPF

EFVal = -3 # electric field normalized to Ec
ZeffVal = 1.5
alphaVal = 0.05
tVal = 1*tMax


# number of grid points used for performing integrals
numpInt = 100
numxiInt = 80
pgridInt = np.linspace(pMin,pMax,numpInt)
xigridInt = np.linspace(xiMin,xiMax,numxiInt)
ggridInt = np.sqrt(pgridInt**2+1)

# parameters used to define the initial electron distribution
DpInit = 0.1 * pMax
DxiInit = 0.1
p0Init = 0.75 * pMax
xi0Init = 0


EFValNorm = ( EFVal - EFMin ) / ( EFMax - EFMin )
ZeffValNorm = ( ZeffVal - ZeffMin ) / ( ZeffMax - ZeffMin )
alphaValNorm = ( alphaVal - alphaMin ) / ( alphaMax - alphaMin )
tValNorm = ( tVal - tMin ) / ( tMax - tMin )

EFavg = (abs(EFMax)+abs(EFMin)) / 2

trainpts = np.loadtxt(train_save_path)
loss = np.loadtxt(loss_save_path)


def pde(inputs, outputs):
    dy_p = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (pMax-pMin)
    dy_xi = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    dy_xixi = dde.grad.hessian(outputs, inputs, i=1, j=1)
    dy_t = dde.grad.jacobian(outputs, inputs, i=0, j=5) / (tMax-tMin)

    pNorm, xi = inputs[:, 0:1], inputs[:, 1:2]
    EFNorm = inputs[:, 2:3]
    ZeffNorm = inputs[:, 3:4]
    alphaNorm = inputs[:, 4:5]

    p = pMin + ( pMax - pMin ) * pNorm
    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    Zeff = ZeffMin + ( ZeffMax - ZeffMin ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

    gamma = tf.sqrt(1+p*p)

    CF = gamma*gamma/p**2
    CB = 0.5*(Zeff+1)*gamma/p

    ElecticFieldTerms = -Ephi * ( xi*dy_p + ((1-xi**2)/p)*dy_xi )
    CollisionalTerms = CF*dy_p - (CB/p**2)*( (1-xi**2)*dy_xixi - 2*xi*dy_xi )
    RadiationTerms = alpha * ( gamma*p*(1-xi**2)*dy_p - xi*(1-xi**2)/gamma*dy_xi )

    DpMax = 0.05*pMax
    StepFunc = 1 - tf.exp(-(p-pMax)**2/DpMax**2)
    loss = StepFunc * (p**2/(1+p**2)) * ( dy_t + ElecticFieldTerms + CollisionalTerms + RadiationTerms )

    return loss


def output_transform(inputs, outputs):
    pNorm, xi = inputs[:, 0:1], inputs[:, 1:2]
    EFNorm = inputs[:, 2:3]
    ZeffNorm = inputs[:, 3:4]
    alphaNorm = inputs[:, 4:5]
    tNorm = inputs[:, 5:6]

    p = pMin + ( pMax - pMin ) * pNorm
    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    t = tMin + ( tMax - tMin ) * tNorm
    
    DProbp = 0.15
    ProbpInit = tf.tanh((p-pRE)/Dp)/DProbp
    
    Probp = ProbpInit + (p-pMin)/(pMax-pMin) * tf.tanh(t) * outputs[:, 0:1]
    Probp = 0.5 * ( 1 + tf.tanh(Probp) )

    return tf.concat(Probp, axis=1)


# Define initial electron distribution
Ixi = np.sqrt(np.pi) / 2 * DxiInit * ( erf((xi0Init+1)/DxiInit) - erf((xi0Init-1)/DxiInit)  )
Ip = np.sqrt(np.pi) / 4 * (2*p0Init**2+DpInit**2) * (DpInit*erf(p0Init/DpInit) + DpInit) + p0Init*DpInit**2/2 * np.exp(-p0Init**2/DpInit**2)
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

    nRE = simps(simps(integrand,pgridInt),xigridInt)
    
    return nRE


def NumberSecnRE(model,EF,Zeff,alpha,tVal,dt):
    EFValNorm = ( EF - EFMin ) / ( EFMax - EFMin )
    ZeffValNorm = ( Zeff - ZeffMin ) / ( ZeffMax - ZeffMin )
    alphaValNorm = ( alpha - alphaMin ) / ( alphaMax - alphaMin )
    tValNorm = ( tVal - tMin ) / ( tMax - tMin )

    gammagrid = np.sqrt(pgridInt**2+1)
    v = pgridInt/gammagrid
    pgridNormInt = ( pgridInt - pMin ) / ( pMax - pMin )

    xi1 = -np.sqrt((gammagrid-1)/(gammagrid+1))
    
    EFValVec = EFValNorm * np.ones(numpInt)
    ZeffValVec = ZeffValNorm * np.ones(numpInt)
    alphaValVec = alphaValNorm * np.ones(numpInt)
    tValVec = tValNorm * np.ones(numpInt)

    X = np.zeros([numpInt,6])

    X[:,0] = pgridNormInt
    X[:,1] = xi1
    X[:,2] = EFValVec
    X[:,3] = ZeffValVec
    X[:,4] = alphaValVec
    X[:,5] = tValVec

    y_2 = model.predict(X)
    P_2 = y_2[:,0]

    
    integrand = dt / 2 / CouLog0 * v / (gammagrid-1)**2 * P_2
    nRE = simps(integrand,pgridInt)
    
    return nRE


def ComputenREaval(model,pgridInt,xigridInt,tgrid,EF,Zeff,alpha):
    # Estimate avalanche growth using R-P
    nREseed = np.zeros(numt)
    fe = Computefe(pgridInt,xigridInt)
    for i in range(0,numt):
        nREseed[i] = NumbernRE(model,EF,Zeff,alpha,tgrid[i],fe)

    dt = tgrid[1] - tgrid[0]
    nSecvstRP = np.zeros(numt)
    for i in range(0,numt):
        #nSecvstRP[i] = NumbernRE(model,EF,Zeff,alpha,tgrid[i],feSec)
        nSecvstRP[i] = NumberSecnRE(model,EF,Zeff,alpha,tgrid[i],dt)

    #                  time, generation
    nREgen = np.zeros([numt,numt])

    # Set time evolution of 0th generation (seed)
    for i in range(0,numt):
        nREgen[i,0] = nREseed[i]

    nREvstRP = np.zeros(numt)
    nREvstRP[0] = nREseed[0]

    for j in range(1,numt): # generation
        for k in range(0,j): # evaluate current number of REs
            nREvstRP[j] += nREgen[j,k]
            
        for i in range(j,numt): # time, compute time evolution of each generation
            nREgen[i,j] = nREvstRP[j] * nSecvstRP[i-j]


    nREvstRP = np.zeros(numt)
    nREvstRP[0] = nREseed[0]
    for i in range(1,numt): # time
        for j in range(0,i+1): # generation
            nREvstRP[i] += nREgen[i,j]

    
    return nREvstRP, nREgen


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


# Load training points
xtrainpts = trainpts[:,0]
ytrainpts = trainpts[:,1]


# Load loss history
steps = loss[:,0]
lossPDE = loss[:,1]
lossBC = loss[:,2]
testPDE = loss[:,3]
testBC = loss[:,4]


################################################
# Define and load model.
# Must have identical architecture as REdecay.py
################################################
#                              pMin, xiMin, EFmin, Zeffmin, alphamin
geom = dde.geometry.Hypercube([0, xiMin, 0, 0, 0], [1, xiMax, 1, 1, 1])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(geom, temporal_domain)

net = dde.maps.FNN([6] + [64] * 4 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

losses = []

data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    losses,
)

model = dde.Model(data, net)
model.compile("L-BFGS-B")
model.restore(save_path = ckpt_save_path, verbose=1)


###################################################
# Predict RPF as a function p and \xi at given time
###################################################
nump = 100
numxi = 50

pgrid = np.linspace(pMin,pMax,nump)
xigrid = np.linspace(xiMin,xiMax,numxi)

pgridNorm = (pgrid - pMin) / (pMax-pMin)

ggrid = np.sqrt(1+pgrid**2)

pnewNorm, xinew = np.meshgrid(pgridNorm,xigrid)
X = np.vstack((np.ravel(pnewNorm), np.ravel(xinew))).T
EFValVec = EFValNorm * np.ones([nump*numxi,1])
ZeffValVec = ZeffValNorm * np.ones([nump*numxi,1])
alphaValVec = alphaValNorm * np.ones([nump*numxi,1])
tValVec = tValNorm * np.ones([nump*numxi,1])
X = np.hstack((X, EFValVec))
X = np.hstack((X, ZeffValVec))
X = np.hstack((X, alphaValVec))
X = np.hstack((X, tValVec))

# Predict RFP
y_pred = model.predict(X)
Pnew = y_pred[:,0]
Pnew = Pnew.reshape(pnewNorm.shape)

# Compute residual
res_pred = model.predict(X,operator=pde)
resnew = res_pred[:,0]
resnew = resnew.reshape(pnewNorm.shape)


# Computes evolution of primary n_RE
numt = 10
tgrid = np.linspace(tMin,tMax,numt)
nREvst = np.zeros(numt)
fe = Computefe(pgridInt,xigridInt)
for i in range(0,numt):
    nREvst[i] = NumbernRE(model,EFVal,ZeffVal,alphaVal,tgrid[i],fe)


# Computes n_RE with the avalanche source term
nREvstRP, nREgen = ComputenREaval(model,pgridInt,xigridInt,tgrid,EFVal,ZeffVal,alphaVal)


#########################
# Plot various quantities
#########################
plt.rcParams.update({'font.size': 18})
fig1, ax1 = plt.subplots(num=1,nrows=1,ncols=1, clear=True)
fig1.set_tight_layout(True)

cs1 = ax1.contourf(1e-6*mecSQ*(ggrid-1), xigrid, Pnew, levels=50, cmap='jet')
ax1.contour(1.e-6*mecSQ*(ggrid-1), xigrid, Pnew, [0.5], colors='black')

fig1.colorbar(cs1,ax=ax1)
ax1.set_ylabel("$\\xi$")
ax1.set_xlabel("Energy [MeV]")
ax1.set_title("RPF")


fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)

cs2 = ax2.contourf(1.e-6*mecSQ*(ggrid-1), xigrid, resnew, 50,cmap='jet')

fig2.colorbar(cs2,ax=ax2)
ax2.set_ylabel("$\\xi$")
ax2.set_xlabel("Energy [MeV]")
ax2.set_title("residual of RPF")


fig3, ax3 = plt.subplots(num=3,nrows=1,ncols=1, clear=True)
fig3.set_tight_layout(True)

ax3.plot(tgrid, nREvstRP/nREvstRP[0], label='PINN: Total', linestyle='-',color='blue',linewidth=2)
ax3.plot(tgrid, nREgen[:,0]/nREgen[0,0], label='PINN: Primary', linestyle='--',color='black',linewidth=2)

for i in range(1,numt):
    ax3.plot(tgrid, nREgen[:,i], label='$n_{RE}$', linestyle='--',linewidth=2)

ax3.set_ylabel("$n_{RE}(t)$")
ax3.set_xlabel("$t/\\tau_c$")
ax3.set_yscale("log")
#ax3.legend()


fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
fig7.set_tight_layout(True)

ptrainpts = pMin + (pMax-pMin) * xtrainpts
gtrainpts = np.sqrt(ptrainpts**2+1)
Energytrainpts = mecSQ * (gtrainpts-1)

ax7.scatter(1.e-6*Energytrainpts, ytrainpts,s=0.0001,color='black')
ax7.set_ylabel("$\\xi$")
ax7.set_xlabel("Energy [MeV]")
ax7.set_title("Training Points")


fig8, ax8 = plt.subplots(num=8,nrows=1,ncols=1, clear=True)
fig8.set_tight_layout(True)

# Removes redundant values in test training array
testPDEClean = np.copy(testPDE)
for i in range(1,len(lossPDE)):
    if np.isclose(lossPDE[i], testPDE[i]):
        testPDEClean[i] = 'nan'

testBCClean = np.copy(testBC)
for i in range(1,len(lossBC)):
    if np.isclose(lossBC[i], testBC[i]):
        testBCClean[i] = 'nan'

ax8.plot(1e-3*steps, lossPDE, label='training PDE', linestyle='-',color='blue',linewidth=2)
ax8.plot(1e-3*steps, lossBC, label='training BC', linestyle='-',color='red',linewidth=2)
ax8.plot(1e-3*steps, testPDEClean, 'xb', label='test PDE')
ax8.plot(1e-3*steps, testBCClean, 'xr', label='test BC')

ax8.set_xlabel("Thousands of epochs")
ax8.set_title("Loss History")
ax8.set_yscale("log")
ax8.set_ylim(1e-9, 1e3)
ax8.legend(loc='upper right')





plt.show(block=True)
