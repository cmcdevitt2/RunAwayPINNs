'''
This script uses a tailored physics-informed neural network to solve the adjoint of the Fokker-Planck
Equation. Details about this script is provied in https://arxiv.org/abs/2403.04948
'''

#importing relevant libraries


import deepxde as dde #DeepXDE
import numpy as np
from deepxde.backend import tf


dde.config.set_default_float("float64") # DeepXDE requires this to be set
tf.random.set_random_seed(1234) # Setting fixed random number generator seed

# Physical constants
mecSQ = 511e3 # electron rest mass in units eV

# Numerical parameters
epochsADAM = 5000 # number of epochs for the adam optimizer
epochsBFGS = 10000 # number of epochs for each LBFGS-B optimizer training period
NumBFGS = 100 # number of LBFGS-B training periods
lr = 5.e-4 # learning rate for the adam optimizer
pts = 100000 # number of points sampled in the domain


# Setting ranges of physics parameters that the PINN will learn

# electric field normalized to connor-hastie electric field
# Electric field is in negative units
EFMin = -10
EFMax = -1

# effective charge
ZeffMin = 1
ZeffMax = 10

# synchrotron radiation strength
alphaMin = 0
alphaMax = 0.2

# Energy
EnergyMaxeV = 5e6
EnergyMineV = 1e4

# Lorentz factor
gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

# momentum normalized to m_e*c
pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)

# pitch-angle
xiMax = 1
xiMin = -1

# Function that defines the PDE to be learned by the PINN
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
    loss = (p**2/(1+p**2)) * ( ElecticFieldTerms + CollisionalTerms + RadiationTerms ) #/ -Ephi

    return loss

# Embedding physics constraints by adding an additional layer in the neural network
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

def main():
    # defining geometry for the PINN
    geom = dde.geometry.Hypercube([pMin, xiMin, 0, 0, 0], [pMax, xiMax, 1, 1, 1])

    # construct neural network
    numNeurons = 32 # number of neurons for each hidden layer
    numLayers  = 4  # number of hidden layers
    net = dde.maps.FNN([5] + [numNeurons] * numLayers + [1], "tanh", "Glorot normal") # 5 corresponds to the number of dimensions

    # apply additional layer on network
    net.apply_output_transform(output_transform)
    
    # boundary function to enforce a dirchlet boundary condition at p_max
    def boundary(inputs, on_boundary):

        # unfolding the inputs
        p = inputs[0]
        xi = inputs[1]
        EFNorm = inputs[2] # normalized electric field to be between 0 and 1
        alphaNorm = inputs[4] # normalied synchrotron strength to be between 0 and 1

        # un-normalizing inputs
        Ephi = EFMin + ( EFMax - EFMin ) * EFNorm
        alpha = alphaMin + ( alphaMax - alphaMin ) * alphaNorm

        # energy flux equation
        Up = xi*Ephi - (gMax**2/pMax**2) - alpha*gMax*pMax*(1-xi**2)

        # return boolean if p = p_max and U_p > 0
        return on_boundary and np.isclose(p, pMax) and Up > 0
    
    # applying the Dirchlet BC at p_max
    bc_pMax = dde.DirichletBC(
        geom, # geometry to apply it to
        lambda x: 1 + 0*x[:, 1:2], # sets the RPF to be 1 if boundary=True
        boundary,
        component=0
    )
    
    # constructing all losses to be minimized by the PINN (PDE loss is already included)
    # and does not need to be added
    losses = [bc_pMax]

    # Constructing data object which the PINN will use for the model
    data = dde.data.PDE(
        geom,                            # geometry
        pde,                             # pde
        losses,                          # loss terms
        num_domain=pts,                  # number of points in the domain
        num_boundary=10000,              # number of points on the boundary of the geometry
        num_test=pts,                    # number of test points on the domain
        train_distribution='Hammersley', # training point distribution
    )

    # constructing PINN model
    model = dde.Model(data, net)

    # Compiling the model with the adam optimizer
    model.compile("adam", lr=lr)
    
    # Training the PINN with the adam optimizer
    losshistory, train_state = model.train(epochs=epochsADAM, model_save_path = './model/model.ckpt')
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)

    # Looping through each LBFGS-B optimizer training period
    for i in range(0,NumBFGS):
        
        # compiling the model with the L-BFGS-B optimizer
        model.compile("L-BFGS-B")

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



        # # routine that adds additional points at maximum of the residual of the PDE
        numPointsAdd = 100000 # number of additional points to add
        xpp = geom.random_points(numPointsAdd)

        # predict the model
        f = model.predict(xpp, operator=pde)

        # evaluate residual
        err_eq_f = np.absolute(f)

        # Add additional points
        for i in range(0,10):
            x_id = np.argmax(err_eq_f)          # index to add points
            data.add_anchors(xpp[x_id])         # add additional points
            err_eq_f = np.delete(err_eq_f,x_id) # delete previous index

if __name__ == "__main__":
    main()
