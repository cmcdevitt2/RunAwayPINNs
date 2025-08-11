'''
This script uses a tailored physics-informed neural network to solve the adjoint of the Fokker-Planck
Equation. Details about this script is provied in https://doi.org/10.1017/S0022377824000679
'''

#importing relevant libraries
import deepxde as dde             # DeepXDE
import numpy as np                # Numpy
from deepxde.backend import torch # pytorch


dde.config.set_default_float("float64") # DeepXDE requires this to be set for L-BFGS-B
dde.config.set_random_seed(1234)        # Setting fixed random number generator seed

# Physical constants
mecSQ = 511e3 # electron rest mass in units eV

# Numerical parameters
epochsADAM  = 0_000    # number of epochs for the adam optimizer
epochsBFGS  = 10_000    # number of epochs for each LBFGS-B optimizer training period
NumBFGS     = 100       # number of LBFGS-B training periods
lr          = 5.e-4     # learning rate for the adam optimizer
ptsTrain    = 1_000_000 # number of points sampled in the domain
ptsTest     = 1_000_000 # number of test points
ptsBoundary = 10_000    # number of points sampled along domain boundary

'''
Setting ranges of physics parameters that the PINN will learn
'''
# electric field normalized to connor-hastie electric field and in negative units
EFMin, EFMax = -10.0, -1.0

# effective charge
ZeffMin, ZeffMax = 1.0, 10.0

# synchrotron radiation strength
alphaMin, alphaMax = 0.0, 0.2

# Energy
EnergyMineV, EnergyMaxeV = 1e4, 5e6

# Lorentz factor
gMin, gMax = 1 + EnergyMineV / mecSQ, 1 + EnergyMaxeV / mecSQ

# momentum normalized to m_e*c
pMin, pMax = np.sqrt(gMin**2-1), np.sqrt(gMax**2-1)

# pitch-angle
xiMin, xiMax = -1.0, 1.0

# neural network parameters
numNeurons = 64     # number of neurons for each hidden layer
numLayers  = 6      # number of hidden layers
numInputs  = 5      # number of inputs
numOutputs = 1      # number of outputs
activation = 'tanh' # activation function


# Function that defines the PDE to be learned by the PINN
def pde(inputs, outputs):
    
    # auto-differentiation
    P_p    = dde.grad.jacobian(outputs, inputs, i=0, j=0) # j=0 is the first  input (p )
    P_xi   = dde.grad.jacobian(outputs, inputs, i=0, j=1) # j=1 is the second input (xi)
    P_xixi = dde.grad.hessian(outputs , inputs, i=1, j=1) # j=1 is the second input (xi)

    # Unfolding the inputs into its respective parameters 
    # 0=p, 1=xi, 2=EF, 3=Zeff, 4=alpha
    p         = inputs[:, 0:1]
    xi        = inputs[:, 1:2]
    EFNorm    = inputs[:, 2:3] # electric field normalized to be between 0 and 1
    ZeffNorm  = inputs[:, 3:4] # effective charge normalized to be between 0 and 1
    alphaNorm = inputs[:, 4:5] # synchrotron radiation strength normalized to be between 0 and 1

    # Un-normalizing the input parameters
    Ephi  = EFMin    + ( EFMax    - EFMin    ) * EFNorm
    Zeff  = ZeffMin  + ( ZeffMax  - ZeffMin  ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

    # computing lorentz factor
    gamma = torch.sqrt(1+p**2)

    # Collision coefficients
    C_F = (gamma*gamma/p**2)        # collisional drag
    nuD = ((Zeff+1)/2)*(gamma/p**3) # pitch-angle scattering

    # Electric field terms in PDE, acceleration in p and anisotropy in \xi
    ElecticFieldTerms = -Ephi * ( xi*P_p + ((1-xi**2)/p)*P_xi )

    # Collisional terms in PDE
    CollisionalTerms = C_F*P_p - nuD*( (1-xi**2)*P_xixi - 2*xi*P_xi ) 

    # Synchrotron radiation terms in PDE
    RadiationTerms = alpha * ( gamma*p*(1-xi**2)*P_p - xi*(1-xi**2)/gamma*P_xi )

    # Constructing the loss term, scaled by the inverse of the collisional drag to prevent
    # divergence at low momentum
    loss = (1/C_F) * ( ElecticFieldTerms + CollisionalTerms + RadiationTerms )

    return loss

# Embedding physics constraints by adding an additional layer in the neural network
def output_transform(inputs, outputs):

    # Un-folding input parameters required
    # 0=p, 1=xi, 2=EF, 3=Zeff, 4=alpha
    p      = inputs[:, 0:1]
    xi     = inputs[:, 1:2]
    EFNorm = inputs[:, 2:3] # normalized electric field to be between 0 and 1

    # Output of neural network
    PNN = outputs[:,0:1]

    # un-normalizing electric field
    Ephi = EFMin + ( EFMax - EFMin ) * EFNorm

    # computing lorentz factor
    gamma = torch.sqrt(p**2+1)

    # Smoothing factor for heaviside function containing the electric field
    dEphi = 0.1

    # heaviside function for the electric field, vanishes at Ephi = 1 and assymptotes to 1
    Heaviside = 0.5 * ( 1 + torch.tanh((-1-Ephi)/dEphi) )

    # constraining PINN output to vanish for Ephi < 1 and p = pMin
    Pprime = Heaviside*((p-pMin)/(pMax-pMin)) * PNN

    # constraining the PINN to be between 0 and 1
    P = torch.tanh(Pprime**2)

    return P

# boundary function to enforce a dirchlet boundary condition at p_max
def boundary(inputs, on_boundary):

    # unfolding the inputs
    # 0=p, 1=xi, 2=EF, 3=Zeff, 4=alpha
    p         = inputs[0]
    xi        = inputs[1]
    EFNorm    = inputs[2] # normalized electric field to be between 0 and 1
    alphaNorm = inputs[4] # normalied synchrotron strength to be between 0 and 1

    # un-normalizing inputs
    Ephi  = EFMin    + ( EFMax    - EFMin    ) * EFNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

    # energy flux equation
    CFMax = gMax**2/pMax**2 # Collisional drag at p_max
    Up = xi*Ephi - CFMax - alpha*gMax*pMax*(1-xi**2)

    # return boolean if p = p_max and U_p > 0
    return on_boundary and dde.utils.isclose(p, pMax) and Up > 0

# defining geometry for the PINN
#                               p      xi   EFNorm, ZeffNorm, alphaNorm
geom = dde.geometry.Hypercube([pMin, xiMin,   0,       0,         0], # min value
                              [pMax, xiMax,   1,       1,         1]) # max value

# Construct fully-connected neural network
net = dde.maps.FNN([numInputs] + [numNeurons] * numLayers + [numOutputs], activation, "Glorot normal")

# apply physics transform on last layer of network
net.apply_output_transform(output_transform)

# applying the Dirchlet BC at p_max
bc_pMax = dde.DirichletBC(
    geom,                      # geometry to apply it to
    lambda x: 1 + 0*x[:, 1:2], # sets the RPF to be 1 if boundary=True
    boundary,                  # function to apply BC on
    component=0                # Output dimension
)

# constructing all losses to be minimized by the PINN (PDE loss is already included)
# and does not need to be added
losses = [bc_pMax]

def main():
    # Constructing data object which the PINN will use for the model
    data = dde.data.PDE(
        geom,                             # geometry
        pde,                              # pde
        losses,                           # loss terms
        num_domain        = ptsTrain,     # number of points in the domain
        num_boundary      = ptsBoundary,  # number of points on the boundary of the geometry
        num_test          = ptsTest,      # number of test points on the domain
        train_distribution= 'Hammersley', # training point distribution
    )

    
    model = dde.Model(data, net) # constructing PINN model
    model.compile("adam", lr=lr) # Compiling the model with the adam optimizer
    
    # Training the PINN with the adam optimizer and save progress
    losshistory, train_state = model.train(iterations=epochsADAM, model_save_path = './model.ckpt')
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)

    # Looping through each LBFGS-B optimizer training period
    for i in range(0,NumBFGS):
        model.compile("L-BFGS-B")# compiling the model with the L-BFGS-B optimizer

        # Training the model
        model.train_step.optimizer_kwargs = {'options': {'maxcor': 100,
                                                         'ftol': 1.0 * np.finfo(float).eps, # prevents early stopping
                                                         'gtol': 1.0 * np.finfo(float).eps, # prevents early stopping
                                                         'maxiter': epochsBFGS,
                                                         'maxfun':  epochsBFGS,
                                                         'maxls': 200}}

        # Saving loss and model at end of training period
        losshistory, train_state = model.train(model_save_path = './model/model.ckpt')
        dde.saveplot(losshistory, train_state, issave=True, isplot=False)
        
if __name__ == "__main__":
    main()
