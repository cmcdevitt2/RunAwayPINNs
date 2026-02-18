# Import required libraries
import torch
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib import ticker
import skopt
from torch.optim import lr_scheduler
import tqdm

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 25})



# Physical constants used
mecSQ = 510998.95 # electron rest mass [eV]

# Set fixed seed for reproducability/debugging
torch.manual_seed(1234)
np.random.seed(1234)

# Set hardware to train PINN (GPU, Apple Silicon, or CPU)
device = torch.device('cuda' if torch.cuda.is_available()         else \
                    'mps'  if torch.backends.mps.is_available() else \
                    'cpu'
                   )


'''
Proceed with defining functions that will be used 
during PINN training
'''
# Loss error, choose as user sees fit
PINNLoss = torch.nn.MSELoss()

# Fully connected neural network
class FCNN(torch.nn.Module):
    def __init__(self, input_dim, features):
        super().__init__()
        layers = []
        for i, fs in enumerate(features):
            in_dim = input_dim if i == 0 else features[i-1]
            layers.append(torch.nn.Linear(in_dim, fs))
            if i < len(features) - 1:
                layers.append(torch.nn.Tanh())
        self.net = torch.nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, inputs):
        return self.net(inputs)

# Hammersly sampling
def hammersley_sequence(n_samples, dim):
    skip = 0
    if dim == 1:
        sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
    else:
        sampler = skopt.sampler.Hammersly()
        skip = 1
    space = [(0.0, 1.0)] * dim
    points = np.asarray(sampler.generate(space, n_samples + skip)[skip:],dtype=np.float32)
    return points

# Function to train with a PyTorch optimizer
def TrainPyTorch(epochs,optimizer,loss_fn,model,lr,lrVals,lrUpdateEpoch,X_train,print_every):
    t = tqdm.trange(epochs)
    for epoch in t:
        optimizer.zero_grad()
        residual = pde(model,X_train)
        loss = PINNLoss(residual,torch.zeros_like(residual))
        t.set_postfix({"loss": f"{loss.detach():.4e}"})
        t.refresh()
        loss.backward()
        optimizer.step()
        # if epoch%lrUpdateEpoch==0 and lrVals.get_last_lr()[0]>lr:
            # lrVals.step()

'''
Proceed with setting up PDE
'''
EMax, EMin = 5e6, 1e4                               # Energy [eV]
gMax, gMin = EMax/mecSQ+1, EMin/mecSQ+1             # Lorentz factor

# Independent variables for the PDE
pMax , pMin  = np.sqrt(gMax**2-1), np.sqrt(gMin**2-1) # Momentum normalized to m_e*c
xiMax, xiMin = 1                 , -1                 # Pitorch-angle

# Coefficients for PDE
Ephi  = -10.0     # Electric field normalized to E_c (in negative units)
Zeff  =   1.0     # effective charge
alpha =   0.0     # Synchrotron radiation strength

# Physics transform on the outputs of the neural network
def PhysicsOutTransform(inputs, outputs):
    pNorm, xiNorm = inputs.split(1,dim=1)
    PNN           = outputs

    p     = pNorm     * (pMax     - pMin    ) + pMin
    xi    = xiNorm    * (xiMax    - xiMin   ) + xiMin 

    CFStar = gMax**2/pMax**2

    xi_star = (-Ephi - np.sqrt(Ephi**2 - 4*(alpha*gMax*pMax*(-CFStar-alpha*gMax*pMax))))/(2*alpha*gMax*pMax + np.finfo(float).eps)

    dEphi  = 5.0
    H_Ephi = 0.5*(1 + np.tanh((abs(Ephi)-1)/dEphi))
    dxi    = 0.15*H_Ephi + 0.1
    H_xi   = 0.5*(1 - torch.tanh((xi-xi_star)/dxi))

    dEphi  = 0.25
    H_Ephi = 0.5*(1 + np.tanh((abs(Ephi)-1)/dEphi))
   
    pRE    = pNorm*H_xi
    BCpMax = (pMax-p*H_xi)/(pMax-pMin)
    BCpMin = pNorm

    Pprime = H_Ephi*(pRE + BCpMin*BCpMax*PNN)
    dP     = 0.25
    P      = torch.tanh(Pprime**2/dP**2)
    
    return P

# Calculate PDE and residual
def pde(model, inputs):

    pNorm, xiNorm = inputs.split(1,dim=1)

    p     = pNorm     * (pMax     - pMin    ) + pMin
    xi    = xiNorm    * (xiMax    - xiMin   ) + xiMin 

    dp_bar, dxi_bar = (pMax - pMin), (xiMax - xiMin)

    PNN  = model(inputs)
    P    = PhysicsOutTransform(inputs, PNN)
    ones = torch.ones_like(P)
    
    P_p    = torch.autograd.grad(P   ,inputs, grad_outputs=ones, create_graph=True)[0][:,0:1]/dp_bar
    P_xi   = torch.autograd.grad(P   ,inputs, grad_outputs=ones, create_graph=True)[0][:,1:2]/dxi_bar
    P_xixi = torch.autograd.grad(P_xi,inputs, grad_outputs=ones, create_graph=True)[0][:,1:2]/dxi_bar**2

    g = torch.sqrt(1+p**2)

    CF = g**2/p**2
    CB = (Zeff+1)*g/p/2

    Estar = -Ephi *(xi*P_p + ((1-xi**2)/p)*P_xi)
    Cstar = CF*P_p - CB/p**2*((1-xi**2)*P_xixi - 2*xi*P_xi)
    Rstar = alpha*(g*p*(1-xi**2)*P_p - xi*(1-xi**2)/g*P_xi)

    PreFac = 1/CF / np.sqrt(abs(Ephi))
    residual    = (Estar + Cstar + Rstar)*PreFac
    
    return residual


# Construct PINN model
model = FCNN(2, [8,8,8,8,1]).to(device)

'''
Proceed with setting up the PINN
'''
if __name__ == "__main__":
    N = 25000
    X_train = torch.tensor(hammersley_sequence(N,2)).to(device).requires_grad_(True)
    
    
    
    NumParams = sum(p.numel() for p in model.parameters())
    print(f'Construced PINN model with {(NumParams/1e3):.1f}K parameters')
    
    lr = 1e-3
    lrUpdate = 1e3
    NumEpochs = 5000
    print_every = 1e3
    
    
    # optimizer = Soap(model.parameters(), lr)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    TrainPyTorch(NumEpochs,optimizer,pde,model,lr,scheduler,lrUpdate,X_train,print_every)

    Np  = 200
    Nxi = 100
    
    p_grid = np.logspace(np.log10(pMin), np.log10(pMax))
    xi_grid = np.linspace(xiMin, xiMax, Nxi)
    
    pmesh, ximesh = np.meshgrid(p_grid, xi_grid)
    
    p_grid_norm = torch.tensor((pmesh-pMin)/(pMax-pMin), device=device,dtype=torch.float32).reshape(-1,1)
    xi_grid_norm = torch.tensor((ximesh-xiMin)/(xiMax-xiMin), device=device,dtype=torch.float32).reshape(-1,1)
    
    X = torch.cat((p_grid_norm,
                   xi_grid_norm,
                  ),dim=1).requires_grad_(True)
    
    model.eval()
    with torch.no_grad():
        PNN = model(X)
        P = PhysicsOutTransform(X, PNN).cpu().numpy().reshape(pmesh.shape)
    
    g_grid = np.sqrt(1+p_grid**2)
    e_grid = (g_grid-1)*mecSQ
    
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    cf = ax.contourf(e_grid,xi_grid, P, cmap='jet', levels=50)
    plt.xscale('log')
    cb = plt.colorbar(cf, ticks=[0,0.25,0.5,0.75,1])
    
    plt.xlabel('Energy [MeV]')
    plt.ylabel('$\\xi$')
    plt.title('RPF $(P)$')
    plt.savefig("RPF_pred.png")
    
    residual = pde(model,X).detach().cpu().numpy().reshape(pmesh.shape)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    cf = ax.contourf(e_grid,xi_grid, abs(residual), cmap='inferno', levels=50)
    plt.xscale('log')
    cb = plt.colorbar(cf, format='%.2f')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks() 
    
    plt.xlabel('Energy [MeV]')
    plt.ylabel('$\\xi$')
    plt.title('PDE Residual')
    plt.savefig("Res_pred.png")
    
    plt.show()

    
