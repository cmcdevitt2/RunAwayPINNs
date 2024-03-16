'''
This script plots various physics quantities relevant to runaway electrons
from a trained PINN model in RPF.py
'''

# Importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
from scipy import interpolate
from scipy.integrate import simps
import deepxde as dde
from deepxde.backend import tf
from scipy import integrate
from scipy.special import kn


dde.config.set_default_float("float64") # needed from DeepXDE

# Provide path to data directory
save_path_root = "./old_data/test/Oct22_2023/"

# Specify which model to load
ModelToLoad = 515000

# Provide path to model, training distribution and loss history
ckpt_save_path = str(save_path_root) + f"model/model.ckpt-{ModelToLoad}.ckpt"
train_save_path = str(save_path_root) + "data/train.dat"
loss_save_path = str(save_path_root) + "data/loss.dat"

# Physical Constants
clight = 2.99792e10 # speed of lightin cm/s
CLASSICALER = 2.8179e-13 # classical electron radius in units of cm
mecSQ = 510.9989461e3 # electron rest mass in units eV
IA = 0.017045e6 # Alfven current in Amperes

# Define values to plot the RPF at
EFVal = -3 # electric field normalized to Ec
ZeffVal = 1 # effective charge
alphaVal = 0.2 # synchrotron radiation tau_c/tau_s

# Specify Coulog logarithm
CouLog = 15

# Set ranges of training parameters
# !!! MATCH WITH RPF.py !!!
EFMin = -10
EFMax = -1

ZeffMin = 1
ZeffMax = 10

alphaMin = 0
alphaMax = 0.2

EnergyMaxeV = 5e6
EnergyMineV = 1e4

gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)

xiMax = 1
xiMin = -1

# Normalize training parameters to be between 0 and 1
EFValNorm = ( EFVal - EFMin ) / ( EFMax - EFMin )
ZeffValNorm = ( ZeffVal - ZeffMin ) / ( ZeffMax - ZeffMin )
alphaValNorm = ( alphaVal - alphaMin ) / ( alphaMax - alphaMin )

# load training points and loss history
trainpts = np.loadtxt(train_save_path)
loss = np.loadtxt(loss_save_path)


# !!! COPY PDE FROM RPF.py
def pde(inputs, outputs):
    
    # auto-differentiation
    dy_p = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    dy_pp = dde.grad.hessian(outputs, inputs, i=0, j=0)
    dy_xi = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    dy_xixi = dde.grad.hessian(outputs, inputs, i=1, j=1)

    # Unfolding the inputs into its respective parameters 
    p         = inputs[:, 0:1]
    xi        = inputs[:, 1:2]
    EFNorm    = inputs[:, 2:3] # electric field normalized to be between 0 and 1
    ZeffNorm  = inputs[:, 3:4] # effective charge normalized to be between 0 and 1
    alphaNorm = inputs[:, 4:5] # synchrotron radiation strength normalized to be between 0 and 1

    # Un-normalizing the input parameters
    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    Zeff = ZeffMin + ( ZeffMax - ZeffMin ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

    # computing lorentz factor
    gamma = tf.sqrt(1+p*p)

    # Electric field terms in PDE, acceleration in p and anisotropy in \xi
    ElecticFieldTerms = -Ephi * ( xi*dy_p + ((1-xi**2)/p)*dy_xi )

    # Collisional terms in PDE
    C_F = (gamma*gamma/p**2) # collisional drag
    nuD = ((Zeff+1)/2)*(gamma/p**3) # pitch-angle scattering
    CollisionalTerms = C_F*dy_p - nuD*( (1-xi**2)*dy_xixi - 2*xi*dy_xi ) 

    # Synchrotron radiation terms in PDE
    RadiationTerms = alpha * ( gamma*p*(1-xi**2)*dy_p - xi*(1-xi**2)/gamma*dy_xi )

    # Constructing the loss term, scaled by the inverse of the collisional drag to prevent
    # divergence at low momentum
    loss = (p**2/(1+p**2)) * ( ElecticFieldTerms + CollisionalTerms + RadiationTerms )

    return loss

# !!! COPY FROM RPF.py
def output_transform(inputs, outputs):

    # Un-folding input parameters required
    p      = inputs[:, 0:1]
    xi     = inputs[:, 1:2]
    EFNorm = inputs[:, 2:3] # normalized electric field to be between 0 and 1

    # un-normalizing electric field
    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm

    # computing lorentz factor
    gamma = tf.sqrt(p**2+1)

    # Smoothing factor for heaviside function containing the electric field
    dEphi = 0.1

    # heaviside function for the electric field, vanishes at Ephi = 1 and assymptotes to 1
    Heaviside = 0.5 * ( 1 + tf.tanh((-1-Ephi)/dEphi) )

    # constraining PINN output to vanish for Ephi < 1 and p = pMin
    Prob = Heaviside*((p-pMin)/(pMax-pMin)) * outputs[:, 0:1]

    # constraining the PINN to be between 0 and 1
    Prob = tf.tanh(Prob*Prob)

    return tf.concat((Prob), axis=1)

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

# Construct model 
geom = dde.geometry.Rectangle([pMin, xiMin, 0, 0, 0], [pMax, xiMax, 1, 1, 1])


net = dde.maps.FNN([5] + [64] * 6 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

losses = []

data = dde.data.TimePDE(
    geom,
    pde,
    losses,
    #num_domain=0,
    #num_boundary=0,
    #num_test=2**13,
    #anchors = points
)

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
y_pred = model.predict(X)
Pnew = y_pred[:,0]
Pnew = Pnew.reshape(pnew.shape)

# Evaluate residual of the PDE
res_pred = model.predict(X,operator=pde)
resnew = res_pred[:,0]
resnew = resnew.reshape(pnew.shape)

# Evaluate avalanche growth rate across all training parameters
nump2 = 400    # Number of momentum points
numEF2 = 200   # number of normalized electric field points
numZeff2 = 10  # number of effective charge points
numalpha2 = 21 # number of synchrotron radiation points

# Create normalized grids
EFgridNorm2 = np.linspace(0, 1, numEF2)
ZeffgridNorm2 = np.linspace(0, 1, numZeff2)
alphagridNorm2 = np.linspace(0, 1, numalpha2)

# Create un-normalized grids
EFgrid2 = EFMin + ( EFMax - EFMin ) * EFgridNorm2
Zeffgrid2 = ZeffMin + ( ZeffMax - ZeffMin ) * ZeffgridNorm2
alphagrid2 = alphaMin + ( alphaMax - alphaMin ) * alphagridNorm2

# Initialize avalanche growth rate 
GammaRPsimps = np.zeros([numEF2,numZeff2,numalpha2])

# Evaluate \delta of electric field
dEF = EFgrid2[0] - EFgrid2[1]

for i in range(0,numEF2):
    for j in range(0,numZeff2):
        for k in range(0,numalpha2):

            # Set bounds of integral
            IntEnergyMineV = EnergyMineV
            IntEnergyMaxeV = EnergyMaxeV
            IntgMin = 1 + IntEnergyMineV / mecSQ
            IntgMax = 1 + IntEnergyMaxeV / mecSQ
            IntpMin = np.sqrt(IntgMin**2-1)
            IntpMax = np.sqrt(IntgMax**2-1)

            # Create grid from bounds of integral 
            pgrid2 = np.linspace(IntpMin,IntpMax,nump2)
            ggrid2 = np.sqrt(1+pgrid2**2)

            # Rosenbluth-Putvinski pitch-angle source
            xi1 = - np.sqrt((ggrid2-1)/(ggrid2+1))

            # Mesh grids and set lorentz and velocity grid
            pnew2, xinew = np.meshgrid(pgrid2,xi1)
            pnew2 = pgrid2
            xinew = xi1
            gnew2 = np.sqrt(1+pnew2**2)
            vnew2 = pnew2 / gnew2

            # Create grids of training paramters to predict
            EFValVec2 = EFgridNorm2[i] * np.ones([nump2,1])
            ZeffVec2 = ZeffgridNorm2[j] * np.ones([nump2,1])
            alphaVec2 = alphagridNorm2[k] * np.ones([nump2,1])

            # Create array for PINN to predict
            X2 = np.vstack((pnew2, xinew)).T
            X2 = np.hstack((X2, EFValVec2))
            X2 = np.hstack((X2, ZeffVec2))
            X2 = np.hstack((X2, alphaVec2))

            # Make prediction
            y_pred2 = model.predict(X2)
            Pnew2 = y_pred2[:,0]
            Pnew2 = Pnew2.reshape(pnew2.shape)

            # Evaluate integrand of avalanche growth rate
            integrand = 1/2/CouLog*pgrid2**2 * vnew2 / (gnew2**2-1) / (gnew2-1)**2 * Pnew2

            # Carry out integration
            GammaRPsimps[i,j,k] = simps(integrand,pgrid2)

# Evalate psi_10, avalanche efficiency
NormPsi10RPsimps = np.zeros([numZeff2,numalpha2])
for j in range(0,numZeff2):
    for k in range(0,numalpha2):
        # Take avalanche growth rate at largest electric field
        Gamma0RP = ( GammaRPsimps[1,j,k] - GammaRPsimps[0,j,k] ) / dEF

        # Evaluate avalanche efficiency
        NormPsi10RPsimps[j,k] = 0.5*np.log(10) * IA / Gamma0RP


# Find avalanche threshold
EavRPsimps = np.zeros([numZeff2,numalpha2])
Ethres = 2.e-3 # Threshold of avalanche growth rate to be considerered, requiried
               # since PINN avalanche growth rate is positive definite

for j in range(0,numZeff2):
    for k in range(0,numalpha2):
        
        # Find where avalanche growth rate is greater than threshold
        IndexELessThanEthres = np.where(GammaRPsimps[:,j,k]<Ethres)[0][0]
        
        # get avalanche threshold electric field 
        EavRPsimps[j,k] = abs(EFgrid2[IndexELessThanEthres])
        

Pnew2 = y_pred2[:,0]
Pnew2 = Pnew2.reshape(pnew2.shape)

SRPintegrand = pgrid2**2 * vnew2 / (gnew2**2-1) / (gnew2-1)**2 * Pnew2


#########################
# Plot various quantities
#########################

# RPF at specified parameters
plt.rcParams.update({'font.size': 18})
fig1, ax1 = plt.subplots(num=1,nrows=1,ncols=1, clear=True)
fig1.set_tight_layout(True)
cs1 = ax1.contourf(mecSQ*(ggrid-1), xigrid, Pnew, levels=50, cmap='jet')
fig1.colorbar(cs1,ax=ax1)
ax1.set_xscale("log")
ax1.set_ylabel("$\\xi$")
ax1.set_xlabel("Energy [eV]")
ax1.set_title("RPF")

# Residual
fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)
cs2 = ax2.contourf(mecSQ*(ggrid-1), xigrid, resnew, 50,cmap='jet')
fig2.colorbar(cs2,ax=ax2)
ax2.set_xscale("log")
ax2.set_ylabel("$\\xi$")
ax2.set_xlabel("Energy [eV]")
ax2.set_title("residual of RPF")


# Avalanche threshold electric field
fig4, ax4 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig4.set_tight_layout(True)
cs4 = ax4.contourf(Zeffgrid2, alphagrid2, EavRPsimps.T, 50, cmap='jet')
fig4.colorbar(cs4,ax=ax4)
ax4.set_ylabel("$\\alpha$")
ax4.set_xlabel("$Z_{eff}$")
ax4.set_title("$E_{av}/E_c$")

# Avalanche efficiency
fig5, ax5 = plt.subplots(num=5,nrows=1,ncols=1, clear=True)
fig5.set_tight_layout(True)
ax5.plot(Zeffgrid2, NormPsi10RPsimps[:,0]/1.e6, '-k', linewidth = 2)
ax5.set_ylabel("$2\\pi \\psi_{10}/\\mu_0 R_0$ [MA]")
ax5.set_xlabel("$Z_{eff}$")
ax5.set_title("$2\\pi \\psi_{10}/\\mu_0 R_0$ [MA]")

# Benchmark data from literature
Fig7bMcDevitt_2018_EF = [1.75, 2, 2.25, 2.5, 2.75, 3]
Fig7bMcDevitt_2018_gav_Moller = [0.007165530345367155, 0.022686426125567732, 0.032885477102533096, 0.04200493817125091, 0.05073909995224403, 0.059087454403202705]
Fig7bMcDevitt_2018_gav_RP = [0.014954936647123968, 0.027699177987542803, 0.03651025737423413, 0.0447814910025707, 0.05266701890933477, 0.060475629210400644]

Fig4McDevitt_2019_EF = [1.6831807977147843, 3.671641963654552, 5.669782913215883, 7.658972665019637, 9.666157331178418]
Fig4McDevitt_2019_gav_Moller = [ 0.00030487848951421537, 0.0845480526952617, 0.1464544323029529, 0.20643569723787894, 0.267186314150612]

EFBenchmarkMoller = np.concatenate((Fig7bMcDevitt_2018_EF,Fig4McDevitt_2019_EF),axis=0)
gavBenchmarkMoller = np.concatenate((Fig7bMcDevitt_2018_gav_Moller,Fig4McDevitt_2019_gav_Moller),axis=0)

# Plot avalanche growth rate
fig6, ax6 = plt.subplots(num=6,nrows=1,ncols=1, clear=True)
fig6.set_tight_layout(True)
ax6.plot(abs(EFgrid2), GammaRPsimps[:,0,10], '-k', linewidth = 2, label='$Z_{eff} = 1$')
ax6.plot(abs(EFgrid2), GammaRPsimps[:,4,10], '--k', linewidth = 2, label='$Z_{eff} = 5$')
ax6.plot(abs(EFgrid2), GammaRPsimps[:,9,10], '-.k', linewidth = 2, label='$Z_{eff} = 10$')
ax6.plot(EFBenchmarkMoller, gavBenchmarkMoller, 'xb', label='Moller')
#ax6.plot(Fig7bMcDevitt_2018_EF, Fig7bMcDevitt_2018_gav_Moller, 'xb', label='Moller')
ax6.plot(Fig7bMcDevitt_2018_EF, Fig7bMcDevitt_2018_gav_RP, 'xr', label='R-P')
#ax6.plot(Fig4McDevitt_2019_EF, Fig4McDevitt_2019_gav_Moller, 'xb', label='Moller')
ax6.set_ylabel("$\\gamma_{av}$")
ax6.set_xlabel("$E/E_c$")
ax6.legend()

# Training point distribution
fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
fig7.set_tight_layout(True)
ax7.scatter(xtrainpts, ytrainpts,s=0.01,color='black')
ax7.set_ylabel("$y$")
ax7.set_xlabel("$x$")
ax7.set_title("Training Points")
ax7.axis("scaled")

# Loss history
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

#plt.pause(1.e-4)
plt.show(block=True)
