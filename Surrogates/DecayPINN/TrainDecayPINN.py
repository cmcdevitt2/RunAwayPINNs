##############################################################################
# Solves the time dependent adjoint of the relativistic Fokker-Planck equation
# for a range of electric field strengths, Z_eff, and synchrotron radiation
# Details about this script is provided in https://doi.org/10.1063/5.0253370
##############################################################################

# Importing relevant libraries
import deepxde as dde             # DeepXDE
import numpy as np                # Numpy
from deepxde.backend import torch # pytorch
from scipy.special import kn      # scipy bessel function

dde.config.set_default_float("float64") # DeepXDE requires this to be set for L-BFGS-B
dde.config.set_random_seed(1234)        # Setting fixed random number generator seed

# Physical constants
mecSQ = 511e3 # electron rest mass in units eV

# Numerical parameters
epochsADAM  = 0_000     # number of epochs for the adam optimizer
epochsBFGS  = 10_000    # number of epochs for each LBFGS-B optimizer training period
NumBFGS     = 100       # number of LBFGS-B training periods
lr          = 5.e-4     # learning rate for the adam optimizer
pts         = 200000    # number of training and test points

'''
Setting ranges of physics parameters that the PINN will learn
'''
EnergyMaxeV, EnergyMineV = 5e6                    , 1e4                      # Energy [eV]
gMax       , gMin        = 1 + EnergyMaxeV / mecSQ, 1 + EnergyMineV / mecSQ  # Lorentz factor
pMax       , pMin        = np.sqrt(gMax**2-1)     , np.sqrt(gMin**2-1)       # momentum normalized to m_e*c
xiMax      , xiMin       = 1.0                    , -1.0                     # pitch-angle
tMax       , tMin        = 5.0                    , 0.0                      # time in units of tau_c
EFMax      , EFMin       = -1.0                   , -3.0                     # Electric field normalized to E_c and in negative units
ZeffMax    , ZeffMin     = 2.0                    , 1.0                      # Effective charge
alphaMax   , alphaMin    = 0.1                    , 0.0                      # Synchrotron radiation strength

# sets momentum above which electrons are counted as REs
pRE = 0.25 * pMax
dp  = 0.1*pMax # sets width of transition region in initial RPF
dP = 0.15 # sets width of normalization for PINN in inital condition

# neural network parameters
numNeurons = 64     # number of neurons for each hidden layer
numLayers  = 4      # number of hidden layers
numInputs  = 6      # number of inputs
numOutputs = 1      # number of outputs
activation = 'tanh' # activation function

# Function that defines the PDE to be learned by the PINN
def pde(inputs, outputs):
    
    # auto-differentiation, j=0 is for p, j=1 is for xi, and j=5 is for t
    P_p    = dde.grad.jacobian(outputs, inputs, i=0, j=0) / (pMax-pMin) # jacobian for momentum normalization
    P_xi   = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    P_xixi = dde.grad.hessian(outputs , inputs, i=1, j=1)
    P_t    = dde.grad.jacobian(outputs, inputs, i=0, j=5) / (tMax-tMin) # jacobian for time normalization
    
    # Unfolding the inputs into its respective parameters 
    # 0=p, 1=xi, 2=EF, 3=Zeff, 4=alpha, 5=t (time not used in this function)
    pNorm     = inputs[:, 0:1]
    xi        = inputs[:, 1:2]
    EFNorm    = inputs[:, 2:3]
    ZeffNorm  = inputs[:, 3:4]
    alphaNorm = inputs[:, 4:5]
    
    # Un-normalizing the input parameters
    p     = pMin     + ( pMax     - pMin     ) * pNorm
    Ephi  = EFMin    + ( EFMax    - EFMin    ) * EFNorm
    Zeff  = ZeffMin  + ( ZeffMax  - ZeffMin  ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

    # computing lorentz factor
    gamma = torch.sqrt(1+p*p)

    # Collision coefficients
    CF = gamma*gamma/p**2      # collisional drag
    CB = 0.5*(Zeff+1)*gamma/p  # pitch-angle scattering

    # Computing operator terms in PDE
    Estar = -Ephi * ( xi*P_p + ((1-xi**2)/p)*P_xi )
    Cstar = CF*P_p - (CB/p**2)*( (1-xi**2)*P_xixi - 2*xi*P_xi )
    Rstar = alpha * ( gamma*p*(1-xi**2)*P_p - xi*(1-xi**2)/gamma*P_xi )

    # Create masking function to ingore residual at p_max
    dpMax = 0.05*pMax
    StepFunc = 1 - torch.exp(-(p-pMax)**2/dpMax**2)

    # Constructing the loss term, scaled by the inverse of the collisional drag to prevent
    # divergence at low momentum and the masking function defined above
    loss = StepFunc * (1/CF) * ( P_t + Estar + Cstar + Rstar )

    return loss

# Embedding physics constraints by adding an additional layer in the neural network
def output_transform(inputs, outputs):
    
    # Unfolding the inputs into its respective parameters 
    # 0=p, 1=xi, 2=EF, 3=Zeff, 4=alpha, 5=t (time not used in this function)
    pNorm     = inputs[:, 0:1]
    xi        = inputs[:, 1:2]
    EFNorm    = inputs[:, 2:3]
    ZeffNorm  = inputs[:, 3:4]
    alphaNorm = inputs[:, 4:5]
    tNorm     = inputs[:, 5:6]
    
    # Un-normalizing the input parameters
    p     = pMin     + ( pMax     - pMin     ) * pNorm
    Ephi  = EFMin    + ( EFMax    - EFMin    ) * EFNorm
    Zeff  = ZeffMin  + ( ZeffMax  - ZeffMin  ) * ZeffNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm
    t     = tMin     + ( tMax     - tMin     ) * tNorm

    # Output of neural network
    PNN = outputs[:,0:1]

    
    Pinit = torch.tanh((p-pRE)/dp)/dP

    # constraining PINN to satisfy initial condition and 
    # Dirichlet BC at p_min, where P=0
    Pprime = Pinit + pNorm * torch.tanh(t) * PNN
    
    # constraining the PINN to be between 0 and 1
    P = 0.5 * ( 1 + torch.tanh(Pprime) )

    return P

# boundary function to enforce a dirchlet boundary condition at p_max
def boundary(inputs, on_boundary):
    
    # unfolding the inputs
    # 0=p, 1=xi, 2=EF, 3=Zeff, 4=alpha
    pNorm     = inputs[0]
    xi        = inputs[1]
    EFNorm    = inputs[2]
    alphaNorm = inputs[4]

    # un-normalizing inputs
    p     = pMin     + ( pMax     - pMin     ) * pNorm
    Ephi  = EFMin    + ( EFMax    - EFMin    ) * EFNorm
    alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

    # energy flux equation
    CFMax = gMax**2/pMax**2 # Collisional drag at p_max
    Up = xi*Ephi - (CFMax) - alpha*gMax*pMax*(1-xi**2)
    
    # return boolean if p = p_max and U_p > 0
    return on_boundary and dde.utils.isclose(p, pMax) and Up > 0


# defining geometry for the PINN
#                              p,  xi,   EF, Zeff, alpha
geom = dde.geometry.Hypercube([0, xiMin, 0,   0,    0], # min
                              [1, xiMax, 1,   1,    1]) # max
# Normalzed time range
temporal_domain = dde.geometry.TimeDomain(0,1)

# Constructing spatio-temporal domain
spatio_temporal_domain = dde.geometry.GeometryXTime(geom, temporal_domain)

# Construct fully-connected neural network
net = dde.maps.FNN([numInputs] + [numNeurons] * numLayers + [numOutputs], activation, "Glorot normal")

# apply physics transform on last layer of network
net.apply_output_transform(output_transform)


# applying the Dirchlet BC at p_max
bc_pMax = dde.DirichletBC(
    spatio_temporal_domain,
    lambda x: 1 + 0*x[:, 1:2],
    boundary,
    component=0
)

# constructing all losses to be minimized by the PINN (PDE loss is already included)
# and does not need to be added
losses = [bc_pMax]


def main():
    # Constructing data object which the PINN will use for the model
    data = dde.data.TimePDE(
        spatio_temporal_domain,           # geometry
        pde,                              # pde
        losses,                           # loss terms
        num_domain=pts,                   # number of points in the domain
        num_boundary=round(pts/50),       # number of points on the boundary of the geometry
        num_initial=0,                    # number of initial points
        num_test=pts,                     # number of test points
        train_distribution='Hammersley',  # training point distribution
    )

    loss_weights = [10] + [1] # Weighting PDE loss by 10
    loss = ["MSE"] * 2        # Setting MSE as the metric for each loss term
    
    # constructing PINN model
    model = dde.Model(data, net) 
    
    # Compiling the model with the adam optimizer
    model.compile("adam", lr=lr, loss=loss, loss_weights=loss_weights)

    # Training the PINN with the adam optimizer and save progress
    losshistory, train_state = model.train(iterations=epochsADAM, model_save_path = './model.ckpt')
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)

    # Resample training points every 500 iterations
    resampler = dde.callbacks.PDEPointResampler(period=500)

    # Looping through each LBFGS-B optimizer training period
    for i in range(0,NumBFGS):
        
        # compiling the model with the L-BFGS-B optimizer
        model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)

        # Setting optimizer settings
        model.train_step.optimizer_kwargs = {'options': {'maxcor': 100,
                                                         'ftol': 1.0 * np.finfo(float).eps,
                                                         'gtol': 1.0 * np.finfo(float).eps,
                                                         'maxiter': epochsBFGS,
                                                         'maxfun':  epochsBFGS,
                                                         'maxls': 200}}

        # Train model and save loss and model at end of training period
        losshistory, train_state = model.train(model_save_path = './model.ckpt',callbacks=[resampler])
        dde.saveplot(losshistory, train_state, issave=True, isplot=False)

        
        # Residual based adaptive ressampling training points 
        # [https://doi.org/10.1016/j.cma.2022.115671]
        
        k=1 # increase to add more adaptivity
        c=1 # increase to make distribution of training points more uniform
        FracPts = 0.1 # add points in increments to avoid running out of memory
        NumPtsToAdd = round(FracPts*pts)

        xpp = spatio_temporal_domain.random_points(25*NumPtsToAdd)
        ftmp = np.abs(model.predict(xpp, operator=pde)).astype(np.float64)
        f = ftmp
        err_eq = np.power(f, k) / np.power(f, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        xpp_ids = np.random.choice(a=len(xpp), size=NumPtsToAdd, replace=False, p=err_eq_normalized)
        xpp_selected = xpp[xpp_ids]
        data.replace_with_anchors(xpp_selected)

        for i in range(0, round(1/FracPts)-1):
            xpp = spatio_temporal_domain.random_points(25*NumPtsToAdd)
            ftmp = np.abs(model.predict(xpp, operator=pde)).astype(np.float64)
            f = ftmp
            err_eq = np.power(f, k) / np.power(f, k).mean() + c
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            xpp_ids = np.random.choice(a=len(xpp), size=NumPtsToAdd, replace=False, p=err_eq_normalized)
            xpp_selected = xpp[xpp_ids]
            data.add_anchors(xpp_selected)

if __name__ == "__main__":
    main()
