import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
from scipy import interpolate
from scipy.integrate import simps
import deepxde as dde
from deepxde.backend import tf
from scipy import integrate
from scipy.special import kn


dde.config.set_default_float("float64")

save_path_root = "./old_data/test/Oct22_2023/"
#save_path_root = "./"

ckpt_save_path = str(save_path_root) + "model/model.ckpt-515005.ckpt"
#solution_save_path = str(save_path_root) + "solution0_coarse.dat"
train_save_path = str(save_path_root) + "data/train.dat"
loss_save_path = str(save_path_root) + "data/loss.dat"

clight = 2.99792e10 # speed of lightin cm/s
CLASSICALER = 2.8179e-13 # classical electron radius in units of cm
mecSQ = 510.9989461e3 # electron rest mass in units eV
IA = 0.017045e6 # Alfven current in Amperes

EFVal = -3 # electric field normalized to Ec
ZeffVal = 1 # effective charge
alphaVal = 0.2 # synchrotron radiation tau_c/tau_s

T0 = 0 # Plasma temperature in eV
CouLog = 15

EFMin = -10
EFMax = -1

ZeffMin = 1
ZeffMax = 10

alphaMin = 0
alphaMax = 0.2

EFValNorm = ( EFVal - EFMin ) / ( EFMax - EFMin )
ZeffValNorm = ( ZeffVal - ZeffMin ) / ( ZeffMax - ZeffMin )
alphaValNorm = ( alphaVal - alphaMin ) / ( alphaMax - alphaMin )

EnergyMaxeV = 5e6
EnergyMineV = 1e4
gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)
xiMax = 1
xiMin = -1

#soln = np.loadtxt(solution_save_path)
trainpts = np.loadtxt(train_save_path)
loss = np.loadtxt(loss_save_path)



def pde(inputs, outputs):
    dy_p = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    dy_pp = dde.grad.hessian(outputs, inputs, i=0, j=0)
    dy_xi = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    dy_xixi = dde.grad.hessian(outputs, inputs, i=1, j=1)

    p, xi, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5]

    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    Zeff = ZeffMin + ( ZeffMax - ZeffMin ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    
    Te = T0

    gamma = tf.sqrt(1+p*p)
    CA = (Te/mecSQ) * gamma*gamma*gamma/p/p/p
    dCAdp = -3 * (Te/mecSQ) * gamma/p/p/p/p

    ElecticFieldTerms = -Ephi * ( xi*dy_p + ((1-xi**2)/p)*dy_xi )
    CollisionalTerms = (gamma*gamma/p**2)*dy_p - ((Zeff+1)/2)*(gamma/p**3)*( (1-xi**2)*dy_xixi - 2*xi*dy_xi ) - CA*dy_pp - ( 2*CA/p + dCAdp )*dy_p
    #CollisionalTerms = (gamma*gamma/p**2)*dy_p - ((Zeff+1)/2)*(gamma/p**3)*( (1-xi**2)*dy_xixi - 2*xi*dy_xi )
    RadiationTerms = alpha * ( gamma*p*(1-xi**2)*dy_p - xi*(1-xi**2)/gamma*dy_xi )
    #RadiationTerms = 0

    #loss = (1/(5.e-2+outputs[:, 0:1])) * (p**2/(1+p**2)) * ( ElecticFieldTerms + CollisionalTerms + RadiationTerms )
    loss = (p**2/(1+p**2)) * ( ElecticFieldTerms + CollisionalTerms + RadiationTerms )

    return loss


def output_transform(inputs, outputs):
    p, xi = inputs[:, 0:1], inputs[:, 1:2]
    EFNorm = inputs[:, 2:3]
    #alphaNorm = inputs[:, 4:5]

    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
    #alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    
    gamma = tf.sqrt(p**2+1)
    #theta = vTeOverc**2 / 2

    #Up = xi*Ephi - (gamma*gamma/p**2) - alpha*gamma*p*(1-xi**2)
    #DxiUp = 0.25
    #Up = xi*Ephi - (gMax**2/pMax**2) - alpha*gMax*pMax*(1-xi**2)
    #DxiUp = 0.25*tf.sqrt(Ephi**2+1)
    #XiOut = 0.5 * ( 1 + tf.math.tanh(Up/DxiUp) )
    
    #Dp_t = Dp * (1 + 1e2*t*CA )
    
    DProb = 1
    #DShiftp = Dp*np.arctanh(1-2*DProb*np.sqrt(0.54931)) # Shifts solution so the step function is 0.5 at p=pRE

    #ProbpInit = 0.5 * ( 1 - tf.tanh((pRE-p+DShiftp)/Dp) )
    
    #Dxi = 0.5
    #XiEnv = tf.exp(-(xi-xiMin)**2/Dxi**2)

    dEphi = 0.1
    Heaviside = 0.5 * ( 1 + tf.tanh((-1-Ephi)/dEphi) )
    
    #Prob = 0.5 * ( 1 - tf.tanh((0.5*pMax-p)/Dp) ) + ((p-pMin)/(pMax-pMin)) * ((t-tMin)/(tMax-tMin)) * ((pMax-p*XiEnv)/(pMax-pMin)) * outputs[:, 0:1]
    #Prob = ProbpInit + ((p-pMin)/(pMax-pMin)) * ((t-tMin)/(tMax-tMin)) * ((pMax-p*XiOut)/(pMax-pMin)) * outputs[:, 0:1]
    Prob = Heaviside*((p-pMin)/(pMax-pMin)) * outputs[:, 0:1]

    #DProb = 0.25
    Prob = tf.tanh((Prob/DProb)*(Prob/DProb))

    return tf.concat((Prob), axis=1)



def dP_Ephi(inputs, outputs):
    return dde.grad.jacobian(outputs, inputs, i=0, j=2)


def GetTe(t,TiniteV):
    #Te = (Tinit - Tfinal) * np.exp(-(tMax-t)/DtTQVal) + Tfinal
    TeeV = (TiniteV - Tfinal) * np.exp(-(tMax-t)) + Tfinal
    return TeeV


def GetEphi(t,TiniteV,j0):
    TeeV = GetTe(t,TiniteV)

    # parameters for Spitzer resistivity
    """
    L11 = 0.58 * 32.0/(3.0*np.pi)
    taucOvertauei = (1./3.) * np.sqrt(2./np.pi) * (mecSQ/TiniteV)**1.5
    ecOvera = 4.8032e-9 / aMinor
    eta0 = taucOvertauei * (1.0/(ne0*aMinor**3)) * (IA/ecOvera) * (1.0/L11)
    """
    
    #Einit = -eta0 * jNorm # initial electric field normalized to Ec
    #Ephi = Einit * TiniteV**1.5 / TeeV**1.5
    
    E0 = eta0 * j0
    Ephi = Zeff * T0**1.5 / TeeV**1.5 * E0
    
    return Ephi


"""
def nREintegrand(xi,p,DtTQ,Tinit,j0,t):

    X2 = np.zeros([1,5])
    X2[0,0] = p
    X2[0,1] = xi
    X2[0,2] = DtTQ
    X2[0,3] = Tinit
    X2[0,4] = j0
    X2[0,5] = t

    y_2 = model.predict(X2)
    P_2 = y_2[:,0]
    #P_2 = 1

    Tinit = TinitMineV + ( TinitMaxeV - TinitMineV ) * TinitNorm
    
    integrand = ne0 / np.sqrt(2*np.pi) * (mecSQ/Tinit)**1.5 * p**2 * np.exp(-0.5*mecSQ/Tinit*p**2) * P_2
    
    return integrand
"""


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


nump = 200
numxi = 150

#pgrid = np.linspace(pMin,pMax,nump)
pgrid = np.logspace(np.log10(pMin),np.log10(pMax),nump)
#pgrid = np.logspace(-10,np.log10(pMax),nump)
xigrid = np.linspace(xiMin,xiMax,numxi)

ggrid = np.sqrt(1+pgrid**2)
#print(ggrid)

pnew, xinew = np.meshgrid(pgrid,xigrid)
X = np.vstack((np.ravel(pnew), np.ravel(xinew))).T
EFValVec = EFValNorm * np.ones([nump*numxi,1])
ZeffValVec = ZeffValNorm * np.ones([nump*numxi,1])
alphaValVec = alphaValNorm * np.ones([nump*numxi,1])
X = np.hstack((X, EFValVec))
X = np.hstack((X, ZeffValVec))
X = np.hstack((X, alphaValVec))


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

y_pred = model.predict(X)
Pnew = y_pred[:,0]
Pnew = Pnew.reshape(pnew.shape)

res_pred = model.predict(X,operator=pde)
resnew = res_pred[:,0]
resnew = resnew.reshape(pnew.shape)

# Evaluate integrand for number of REs

nump2 = 400

numEF2 = 200
numZeff2 = 10
numalpha2 = 21
EFgridNorm2 = np.linspace(0, 1, numEF2)
ZeffgridNorm2 = np.linspace(0, 1, numZeff2)
alphagridNorm2 = np.linspace(0, 1, numalpha2)
EFgrid2 = EFMin + ( EFMax - EFMin ) * EFgridNorm2
Zeffgrid2 = ZeffMin + ( ZeffMax - ZeffMin ) * ZeffgridNorm2
alphagrid2 = alphaMin + ( alphaMax - alphaMin ) * alphagridNorm2
GammaRPsimps = np.zeros([numEF2,numZeff2,numalpha2])
NormPsi10RPsimps = np.zeros([numEF2,numZeff2,numalpha2])
print("Zeff = " + str(Zeffgrid2))
print("alpha = " + str(alphagrid2))

dEF = EFgrid2[0] - EFgrid2[1]

#print("dEF = " + str(dEF))

for i in range(0,numEF2):
    for j in range(0,numZeff2):
        for k in range(0,numalpha2):
            IntEnergyMineV = EnergyMineV
            IntEnergyMaxeV = EnergyMaxeV
            IntgMin = 1 + IntEnergyMineV / mecSQ
            IntgMax = 1 + IntEnergyMaxeV / mecSQ
            IntpMin = np.sqrt(IntgMin**2-1)
            IntpMax = np.sqrt(IntgMax**2-1)

            #pgrid2 = np.logspace(np.log10(IntpMin),np.log10(IntpMax),nump2)
            pgrid2 = np.linspace(IntpMin,IntpMax,nump2)
            ggrid2 = np.sqrt(1+pgrid2**2)

            xi1 = - np.sqrt((ggrid2-1)/(ggrid2+1))
        
            pnew2, xinew = np.meshgrid(pgrid2,xi1)
            #pnew2, xinew = np.meshgrid(pgrid2,xigrid)
            pnew2 = pgrid2
            xinew = xi1
            gnew2 = np.sqrt(1+pnew2**2)
            vnew2 = pnew2 / gnew2
        
            #X2 = np.vstack((np.ravel(pnew2), np.ravel(xinew))).T
            X2 = np.vstack((pnew2, xinew)).T
            EFValVec2 = EFgridNorm2[i] * np.ones([nump2,1])
            ZeffVec2 = ZeffgridNorm2[j] * np.ones([nump2,1])
            alphaVec2 = alphagridNorm2[k] * np.ones([nump2,1])
            X2 = np.hstack((X2, EFValVec2))
            X2 = np.hstack((X2, ZeffVec2))
            X2 = np.hstack((X2, alphaVec2))

            y_pred2 = model.predict(X2)
            Pnew2 = y_pred2[:,0]
            Pnew2 = Pnew2.reshape(pnew2.shape)

            #y_pred2 = model.predict(X2,operator=dP_Ephi)
            #dP_Ephinew2 = y_pred2[:,0]
            #dP_Ephinew2 = dP_Ephinew2.reshape(pnew2.shape)

            integrand = 1/2/CouLog*pgrid2**2 * vnew2 / (gnew2**2-1) / (gnew2-1)**2 * Pnew2
            #integrand2 = 1/2/CouLog*pgrid2**2 * vnew2 / (gnew2**2-1) / (gnew2-1)**2 * dP_Ephinew2
        
            #GammaRPsimps[i,j] = simps(simps(integrand,pgrid2),xigrid)
            GammaRPsimps[i,j,k] = simps(integrand,pgrid2)
            #Gamma0RPsimps = simps(integrand2,pgrid2)

        #NormPsi10RPsimps[i,j] = 0.5*np.log(10) * IA / Gamma0RPsimps


# Evalate psi_10
NormPsi10RPsimps = np.zeros([numZeff2,numalpha2])
for j in range(0,numZeff2):
    for k in range(0,numalpha2):
        Gamma0RP = ( GammaRPsimps[1,j,k] - GammaRPsimps[0,j,k] ) / dEF
        #print("Gamma0RP = " + str(Gamma0RP))
        NormPsi10RPsimps[j,k] = 0.5*np.log(10) * IA / Gamma0RP



# Find avalanche threshold
EavRPsimps = np.zeros([numZeff2,numalpha2])
Ethres = 2.e-3
for j in range(0,numZeff2):
    for k in range(0,numalpha2):
        IndexELessThanEthres = np.where(GammaRPsimps[:,j,k]<Ethres)[0][0]
        #print("IndexELessThanEthres =" + str(IndexELessThanEthres))
        EavRPsimps[j,k] = abs(EFgrid2[IndexELessThanEthres])
        #Gamma0RP = ( GammaRPsimps[1,j,k] - GammaRPsimps[0,j,k] ) / dEF
        #NormPsi10RPsimps[j,k] = 0.5*np.log(10) * IA / Gamma0RP

        
#print("psi10 = " + str(NormPsi10RPsimps/1e6))
        
#nump2 = 100
#IntEnergyMineV = 3 * TinitValeV
IntEnergyMineV = EnergyMineV
IntEnergyMaxeV = EnergyMaxeV
IntgMin = 1 + IntEnergyMineV / mecSQ
IntgMax = 1 + IntEnergyMaxeV / mecSQ
IntpMin = np.sqrt(IntgMin**2-1)
IntpMax = np.sqrt(IntgMax**2-1)

#pgrid2 = np.logspace(np.log10(IntpMin),np.log10(IntpMax),nump2)
pgrid2 = np.linspace(IntpMin,IntpMax,nump2)
ggrid2 = np.sqrt(1+pgrid2**2)

pnew2, xinew = np.meshgrid(pgrid2,xigrid)
gnew2 = np.sqrt(1+pnew2**2)
vnew2 = pnew2 / gnew2
        
X2 = np.vstack((np.ravel(pnew2), np.ravel(xinew))).T
EFValVec2 = EFValNorm * np.ones([nump2*numxi,1])
ZeffVec2 = ZeffValNorm * np.ones([nump2*numxi,1])
alphaVec2 = alphaValNorm * np.ones([nump2*numxi,1])
X2 = np.hstack((X2, EFValVec2))
X2 = np.hstack((X2, ZeffVec2))
X2 = np.hstack((X2, alphaVec2))

y_pred2 = model.predict(X2)
Pnew2 = y_pred2[:,0]
Pnew2 = Pnew2.reshape(pnew2.shape)

SRPintegrand = pgrid2**2 * vnew2 / (gnew2**2-1) / (gnew2-1)**2 * Pnew2


#########################
# Plot various quantities
#########################
plt.rcParams.update({'font.size': 18})
fig1, ax1 = plt.subplots(num=1,nrows=1,ncols=1, clear=True)
fig1.set_tight_layout(True)

#cs1 = ax1.contourf(pgrid, xigrid, Pnew, 50,cmap='jet')
cs1 = ax1.contourf(mecSQ*(ggrid-1), xigrid, Pnew, levels=50, cmap='jet')
#cs1 = ax1.contourf(mecSQ*(ggrid-1), xigrid, np.log10(Pnew), levels=50, cmap='jet')

fig1.colorbar(cs1,ax=ax1)
ax1.set_xscale("log")
ax1.set_ylabel("$\\xi$")
#ax1.set_xlabel("$p$")
ax1.set_xlabel("Energy [eV]")
ax1.set_title("RPF")
#ax1.set_xlim(100e3, max(mecSQ*(ggrid-1)))
#ax1.axis("scaled")
#fig.savefig("figure")


fig2, ax2 = plt.subplots(num=2,nrows=1,ncols=1, clear=True)
fig2.set_tight_layout(True)

cs2 = ax2.contourf(mecSQ*(ggrid-1), xigrid, resnew, 50,cmap='jet')
#cs2 = ax2.contourf(mecSQ*(ggrid-1), xigrid, np.log10(abs(resnew)/(1.e-6+Pnew)), 50,cmap='jet')

fig2.colorbar(cs2,ax=ax2)
ax2.set_xscale("log")
ax2.set_ylabel("$\\xi$")
#ax2.set_xlabel("$p$")
ax2.set_xlabel("Energy [eV]")
ax2.set_title("residual of RPF")
#ax2.set_xlim(100e3, max(mecSQ*(ggrid-1)))
#ax2.axis("scaled")
#fig.savefig("figure")


fig4, ax4 = plt.subplots(num=4,nrows=1,ncols=1, clear=True)
fig4.set_tight_layout(True)

cs4 = ax4.contourf(Zeffgrid2, alphagrid2, EavRPsimps.T, 50, cmap='jet')

fig4.colorbar(cs4,ax=ax4)
ax4.set_ylabel("$\\alpha$")
ax4.set_xlabel("$Z_{eff}$")
#ax4.set_yscale("log")
ax4.set_title("$E_{av}/E_c$")
#ax4.axis("scaled")
#ax4.set_ylim(1.e-10, 2.e-2)
#fig.savefig("figure")


fig5, ax5 = plt.subplots(num=5,nrows=1,ncols=1, clear=True)
fig5.set_tight_layout(True)

#cs5 = ax5.contourf(Zeffgrid2, alphagrid2, NormPsi10RPsimps.T/1.e6, 50, cmap='jet')
ax5.plot(Zeffgrid2, NormPsi10RPsimps[:,0]/1.e6, '-k', linewidth = 2)

#fig5.colorbar(cs5,ax=ax5)
#ax5.set_ylabel("$\\alpha$")
ax5.set_ylabel("$2\\pi \\psi_{10}/\\mu_0 R_0$ [MA]")
ax5.set_xlabel("$Z_{eff}$")
#ax5.set_yscale("log")
ax5.set_title("$2\\pi \\psi_{10}/\\mu_0 R_0$ [MA]")
#ax5.axis("scaled")
#ax5.set_ylim(1.e-10, 2.e-2)
#fig.savefig("figure")



Fig7bMcDevitt_2018_EF = [1.75, 2, 2.25, 2.5, 2.75, 3]
Fig7bMcDevitt_2018_gav_Moller = [0.007165530345367155, 0.022686426125567732, 0.032885477102533096, 0.04200493817125091, 0.05073909995224403, 0.059087454403202705]
Fig7bMcDevitt_2018_gav_RP = [0.014954936647123968, 0.027699177987542803, 0.03651025737423413, 0.0447814910025707, 0.05266701890933477, 0.060475629210400644]

Fig4McDevitt_2019_EF = [1.6831807977147843, 3.671641963654552, 5.669782913215883, 7.658972665019637, 9.666157331178418]
Fig4McDevitt_2019_gav_Moller = [ 0.00030487848951421537, 0.0845480526952617, 0.1464544323029529, 0.20643569723787894, 0.267186314150612]

EFBenchmarkMoller = np.concatenate((Fig7bMcDevitt_2018_EF,Fig4McDevitt_2019_EF),axis=0)
gavBenchmarkMoller = np.concatenate((Fig7bMcDevitt_2018_gav_Moller,Fig4McDevitt_2019_gav_Moller),axis=0)

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
#ax6.set_yscale("log")
#ax6.set_title("$n_{RE}$")
#ax6.axis("scaled")
#ax6.set_ylim(1.e-10, 2.e-2)
#fig.savefig("figure")
ax6.legend()


fig7, ax7 = plt.subplots(num=7,nrows=1,ncols=1, clear=True)
fig7.set_tight_layout(True)

ax7.scatter(xtrainpts, ytrainpts,s=0.01,color='black')
ax7.set_ylabel("$y$")
ax7.set_xlabel("$x$")
ax7.set_title("Training Points")
ax7.axis("scaled")
#fig.savefig("figure")


fig8, ax8 = plt.subplots(num=8,nrows=1,ncols=1, clear=True)
fig8.set_tight_layout(True)

ax8.plot(steps/1e3, lossPDE, label='PDE training loss', linestyle='-',color='blue',linewidth=2)
ax8.plot(steps/1e3, lossBC, label='BC training loss', linestyle='-',color='red',linewidth=2)
ax8.plot(steps/1e3, testPDE, label='PDE test loss', linestyle='--',color='blue',linewidth=2)
ax8.plot(steps/1e3, testBC, label='BC test loss', linestyle='--',color='red',linewidth=2)

#ax8.set_ylabel("$y$")
ax8.set_xlabel("thousands of steps")
ax8.set_title("Loss History")
ax8.set_yscale("log")
ax8.legend()
#fig.savefig("figure")


fig9, ax9 = plt.subplots(num=9,nrows=1,ncols=1, clear=True)
fig9.set_tight_layout(True)

#cs1 = ax1.contourf(pgrid, xigrid, Pnew, 50,cmap='jet')
cs9 = ax9.contourf(mecSQ*(ggrid2-1), xigrid, SRPintegrand, 50, cmap='jet')

fig9.colorbar(cs9,ax=ax9)
ax9.set_xscale("log")
ax9.set_ylabel("$\\xi$")
#ax9.set_xlabel("$p$")
ax9.set_xlabel("Energy [eV]")
ax9.set_title("RPF")
#ax9.set_xlim(IntEnergyMineV, IntEnergyMaxeV)
#ax9.axis("scaled")
#fig.savefig("figure")



#plt.pause(1.e-4)
plt.show(block=True)
