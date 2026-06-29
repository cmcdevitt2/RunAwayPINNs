import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.quasirandom import SobolEngine
import numpy as np
import matplotlib.pyplot as plt
from pytorch_optimizer.optimizer.soap import SOAP as Soap
from torch.autograd import grad
import time
import argparse
import gc
import os
import scipy
import tqdm
dataout=os.path.dirname(os.path.abspath(__file__))+'/'
plt.rcParams.update({'font.size': 20})

# Helper function to replace '.' with 'o' in filename (except extension)
def replace_dot_in_filename(filename):
    """Replace '.' with 'o' in filename, keeping the extension intact."""
    # Split path and filename to only replace dots in the filename, not the directory path
    dirname, basename = os.path.split(filename)
    name, ext = os.path.splitext(basename)
    new_basename = name.replace('.', 'o') + ext
    return os.path.join(dirname, new_basename) if dirname else new_basename

# Set random seed for reproducibility
seed= 1234
torch.manual_seed(seed)
np.random.seed(seed)

# Configuration
dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
# Physical constants and parameters
mecSQ = 511e3  # electron rest mass in units eV

# Domain parameters - 7D version
EnergyMaxeV = 16e6
EnergyMineV = 1e4
gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)
xiMin, xiMax = -1, 1
tMin, tMax = 0, 20
EFMin, EFMax = 1.5, 4.0
ZeffMin, ZeffMax = 1.0, 5.0
alphaMin, alphaMax = 0.05, 0.2
pREMin = np.sqrt((1+1E6/mecSQ)**2-1)
pREMax = 0.85*pMax

print(f"pMax: {pMax}, pMin: {pMin}")
print(f"EF range: {EFMin} to {EFMax}")
print(f"Zeff range: {ZeffMin} to {ZeffMax}")
print(f"alpha range: {alphaMin} to {alphaMax}")

# Training parameters
lr = 5.e-4
epochs = 100_000
SSBroyden_cycles = 200
epochsSSBroyden = 5000 
num_domain_points = 5_000_000  # For 7D
num_test_points = 30_000_000   # For 7D
num_boundary_points = 500_000  # For 7D
T = 15e3
architecture= [64,64,64,64,64,64,1]
numSSBroyden = 3_000


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

def output_transform_7D(inputs, outputs):
    """Output transform function for 7D problem"""
    pNorm, xiNorm, pRENorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs.split(1, dim=1)
    Prob = torch.tanh(pNorm*outputs**2)
    return Prob
def pRE_grad(inputs,outputs):
    pNorm, xiNorm, pRENorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs.split(1, dim=1)    
    ones = torch.ones_like(outputs)
    dy_P = grad(outputs, inputs, grad_outputs=ones,create_graph=True)[0]
    dy_pRE = dy_P[:, 2:3]/(pREMax-pREMin)
    return dy_pRE
def pde_residual_7D(inputs, outputs):
    """PDE residual function for 7D problem"""
    pNorm, xiNorm, pRENorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs.split(1, dim=1)
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    EFval = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin
    
    ones = torch.ones_like(outputs)
    dy_P = grad(outputs, inputs, grad_outputs=ones,create_graph=True)[0]
    dy_p = dy_P[:, 0:1]/(pMax-pMin)
    dy_xi = dy_P[:,1:2]/(xiMax-xiMin)
    dy_t = dy_P[:,3:4]/(tMax-tMin)
    dy_xixi = grad(dy_xi, inputs,grad_outputs=ones, create_graph=True)[0][:, 1:2]/(xiMax-xiMin)
    gamma = torch.sqrt(1 + p * p)
    
    Ephi = -EFval
    
    ElectricFieldTerms = -Ephi * (xi * dy_p + ((1 - xi**2) / p) * dy_xi)
    CollisionalTerms = (gamma * gamma / p**2) * dy_p - ((Zeff + 1) / 2) * (gamma / p**3) * ((1 - xi**2) * dy_xixi - 2 * xi * dy_xi)
    RadiationTerms = alpha * (gamma * p * (1 - xi**2) * dy_p - xi * (1 - xi**2) / gamma * dy_xi)

    loss =(p**2/(1+p**2))*(dy_t + ElectricFieldTerms + CollisionalTerms + RadiationTerms)/(alpha/alphaMin)
    return loss


def plot_distribution_7D_gaussian(model,EF_val,Zeff_val,alpha_val):
    start_model_predict_time= time.time()
    nump,numxi,numpRE=200,201,50
    pgrid = np.linspace(0,1, nump)
    # pgrid = np.linspace(pMin,pMax,nump)
    pgridreal = np.linspace(pMin, pMax, nump)
    g_array=np.sqrt(pgridreal**2+1)
    InitgMax = 12.7417 #6MeV
    InitgMin = 6.87084 #3MeV
    # Convert momentum to energy in eV
    p_real_to_eV = np.array([(np.sqrt(p**2 + 1) - 1) * 511e3 for p in pgridreal])
    xigrid = np.linspace(0, 1, numxi)
    # xigrid = np.linspace(xiMin, xiMax, numxi)
    xigrid_real = np.linspace(xiMin, xiMax, numxi)
    
    pREgrid = np.linspace(0,1,numpRE)
    # pREgrid=np.linspace(pREMin,pREMax,numpRE)
    pRE_real=np.linspace(pREMin,pREMax,numpRE)
    EFgrid= np.array([(EF_val-EFMin)/(EFMax-EFMin)])
    Zeffgrid = np.array([(Zeff_val-ZeffMin)/(ZeffMax-ZeffMin)])
    alphagrid = np.array([(alpha_val-alphaMin)/(alphaMax-alphaMin)])
    
    tgrid_real=np.array([0,2,5,10,20])
    tgrid=(tgrid_real-tMin)/(tMax-tMin)
    
    numt= len(tgrid)
    
    # Create meshgrid
    pnew, xinew, pREnew, tnew,EFnew,Zeffnew,alphanew= np.meshgrid(pgrid, xigrid, pREgrid, tgrid,EFgrid,Zeffgrid,alphagrid)
    X = np.vstack((np.ravel(pnew), np.ravel(xinew), np.ravel(pREnew), 
                   np.ravel(tnew), np.ravel(EFnew), np.ravel(Zeffnew),np.ravel(alphanew))).T

    X_torch = torch.tensor(X, dtype=dtype, device=device).requires_grad_(True)
    y_pred = model(X_torch)
    y_pred_transformed = output_transform_7D(X_torch, y_pred)
    dP_dpRE = pRE_grad(X_torch,y_pred_transformed).detach().cpu().numpy().reshape(numxi, nump, numpRE, numt,1,1,1)
    end_model_predict_time = time.time()
    print('model_predict_time',end_model_predict_time-start_model_predict_time)
    pMin_init=np.sqrt(InitgMin**2-1)
    pMax_init=np.sqrt(InitgMax**2-1)
    sigma_p = 3.5
    p_mid= np.sqrt((8/0.511+1)**2-1)
    f_p_RE =  np.array([(np.sqrt(2*np.pi)*sigma_p)**-1*np.exp(-0.5*((p-p_mid)/sigma_p)**2) for p in pgridreal])
    
    for case_name in ['isotropic']:
        fig,ax=plt.subplots(ncols=1,nrows=1,clear=True)
        fig.set_tight_layout(True)
        if case_name=='aligned':
            f_init_xi=np.array([0.5*(1-np.tanh((xi+0.9)/(0.25**10))) for xi in xigrid_real])
            f_p_RE_norm=f_p_RE/((pgridreal*g_array*0.1))
        elif case_name=='isotropic':
            f_init_xi=np.array([0.5*(1-np.tanh((xi-2)/(0.25**10))) for xi in xigrid_real])
            f_p_RE_norm=f_p_RE/((pgridreal*g_array*2))
        elif case_name=='opposed':
            f_init_xi=np.array([0.5*(1+np.tanh((xi-0.9)/(0.25**10))) for xi in xigrid_real])
            f_p_RE_norm=f_p_RE/(pgridreal*g_array*0.1) 
        j=0
        c_list=['r','y','orange','g','b','magenta','black','purple']
        for t in tgrid_real:
            jonta_dir = f'RE_energy_dist_E={"%.2f"%EF_val}_Zeff={"%.2f"%Zeff_val}_alpha={"%.4f"%alpha_val}_isotropic_full/data/energy_distrib_{t*1000:06d}.txt'
            X,y= np.loadtxt(F'{jonta_dir}')
            ax.plot(X,y,c=c_list[j],linestyle='--')#,label='RAMc, t=0$\\tau_c$')
            j+=1

        f_init_xi_norm=f_init_xi
        f_init_ones=np.ones((numxi,nump))
        f_p_RE_norm_reshape=f_p_RE_norm.reshape(1,nump)
        f_init_xi_norm_reshape=f_init_xi_norm.reshape(numxi,1)
        f_init=f_init_ones*f_p_RE_norm_reshape*f_init_xi_norm_reshape
        distribution=f_init.reshape(numxi,nump,1,1,1,1,1) #7D

        pgrid_reshape=pgridreal.reshape(1,nump,1,1,1,1,1)
        y=dP_dpRE*distribution
        p_integrated=scipy.integrate.simpson(y*pgrid_reshape**2,axis=1,x=pgrid_reshape)
        xi_integrated=scipy.integrate.simpson(p_integrated,axis=0,x=xigrid_real)
        pRE_real_to_eV=(np.sqrt((pRE_real.reshape(numpRE,1,1,1))**2+1)-1)*511e3 #5D
        pRE_real_to_eV=pRE_real_to_eV.flatten()
        title_string = r"$E_\Vert = $" + str(EF_val) + r", $Z_{eff}$ = "+ str(Zeff_val) + r", $\alpha = $" + str(alpha_val)
        ax.set_title(title_string)
        y_pinn=-xi_integrated/(2*np.pi)
        ax.plot(pRE_real_to_eV/1e6,y_pinn[:,0,0,0,0].flatten(),c='r',label=f't=0$\\tau_c$')
        ax.plot(pRE_real_to_eV/1e6,y_pinn[:,1,0,0,0].flatten(),c='y',label=f't=2$\\tau_c$')
        ax.plot(pRE_real_to_eV/1e6,y_pinn[:,2,0,0,0].flatten(),c='orange',label=f't=5$\\tau_c$')
        ax.plot(pRE_real_to_eV/1e6,y_pinn[:,3,0,0,0].flatten(),c='g',label=f't=10$\\tau_c$')
        ax.plot(pRE_real_to_eV/1e6,y_pinn[:,4,0,0,0].flatten(),c='b',label=f't=20$\\tau_c$')
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel(r'$p^2f_e$')
        import os
        results_dir = f'{dataout}/results'
        os.makedirs(results_dir, exist_ok=True)
        
        filename1 = f'{results_dir}/{case_name}_4D_EF={EF_val}_Zeff={Zeff_val}_alpha={alpha_val}_gaussian_pytorch.png'
        fig.savefig(replace_dot_in_filename(filename1))
        ax.set_yscale('log')
        ax.set_ylim((1e-5,1e-1))
        filename2 = f'{results_dir}/{case_name}_4D_EF={EF_val}_Zeff={Zeff_val}_alpha={alpha_val}_gaussian_log_pytorch.png'
        fig.savefig(replace_dot_in_filename(filename2))

def plot_RPF_7D(model, pREgrid, tgrid, EFgrid, Zeffgrid, alphagrid, losses, domain_points):
    """Plot RPF for 7D problem"""
    nump, numxi, numpRE, numt, numEF, numZeff, numalpha = 100, 101, 1, 1, 1, 1, 1
    
    pgrid = np.linspace(0, 1, nump)
    pgridreal = np.linspace(pMin, pMax, nump)
    p_real_to_eV = np.array([(np.sqrt(p**2 + 1) - 1) * 511e3 for p in pgridreal])
    
    xigrid = np.linspace(0, 1, numxi)
    xigrid_real=np.linspace(xiMin,xiMax,numxi)
    
    pnew, xinew, pREnew, tnew, EFnew, Zeffnew, alphanew = np.meshgrid(pgrid, xigrid, pREgrid, tgrid, EFgrid, Zeffgrid, alphagrid)
    X = np.vstack((np.ravel(pnew), np.ravel(xinew), np.ravel(pREnew), 
                   np.ravel(tnew), np.ravel(EFnew), np.ravel(Zeffnew), np.ravel(alphanew))).T
    
    X_torch = torch.tensor(X, dtype=dtype, device=device)
    
    with torch.no_grad():
        y_pred = model(X_torch)
        y_pred_transformed = output_transform_7D(X_torch, y_pred)
    
    X_torch_grad = X_torch.detach().requires_grad_(True)
    y_pred_grad = model(X_torch_grad)
    y_pred_transformed_grad = output_transform_7D(X_torch_grad, y_pred_grad)
    pde_res = pde_residual_7D(X_torch_grad, y_pred_transformed_grad)
    
    unew = y_pred_transformed.detach().cpu().numpy().reshape(numxi, nump)
    unew_res = pde_res.detach().cpu().numpy().reshape(numxi, nump)

    print('Test MSE=',f"{np.mean(unew_res**2)}")
    
    fig0, (ax0) = plt.subplots(nrows=1, ncols=1)
    fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig0.set_tight_layout(True)
    
    cs0 = ax0.contourf(p_real_to_eV / 1E6, xigrid_real, unew, 50, cmap='jet')
    # ax0.contour(p_real_to_eV / 1E6, xigrid_real, unew, levels=[0.5], colors='black')
    fig0.colorbar(cs0, ax=ax0)
    
    cs1 = ax1.contourf(p_real_to_eV / 1E6, xigrid_real, unew_res, 50, cmap='jet')
    fig1.colorbar(cs1, ax=ax1)
    
    EF_val = (EFMax-EFMin)*EFgrid+EFMin
    Zeff_val = (ZeffMax-ZeffMin)*Zeffgrid+ZeffMin
    alpha_val = (alphaMax-alphaMin)*alphagrid+alphaMin
    t_val = (tMax-tMin)*tgrid+tMin

    ax0.set_xlabel("Energy [MeV]")
    ax1.set_xlabel("Energy [MeV]")
    ax0.set_ylabel("$\\xi$")
    ax1.set_ylabel("$\\xi$")
    # ax0.set_title(f"RPF "+ rf"t/$\tau_c$={t_val:.2f}")
    # ax1.set_title(f"Residual "+ rf"t/$\tau_c$={t_val:.2f}")
    
    plots_dir = f'{dataout}/plots_7D'
    os.makedirs(plots_dir, exist_ok=True)
    
    filename0 = f'{plots_dir}/RPF_7D_t={t_val:.2f}_EF={EF_val:.2f}_Zeff={Zeff_val:.2f}_alpha={alpha_val:.2f}_pRE={pREgrid}_pytorch.png'
    fig0.savefig(replace_dot_in_filename(filename0), dpi=300, bbox_inches='tight')
    filename1 = f'{plots_dir}/RPF_7D_t={t_val:.2f}_EF={EF_val:.2f}_Zeff={Zeff_val:.2f}_alpha={alpha_val:.2f}_pRE={pREgrid}_pytorch_res.png'
    fig1.savefig(replace_dot_in_filename(filename1), dpi=300, bbox_inches='tight')
    plt.close(fig0)
    plt.close(fig1)
    return


def calculate_density(y_pred_transformed, pgridreal, g_array, xigrid_real, f_p_RE, case_name='isotropic'):
    """Calculate density by integrating over momentum and pitch angle"""
    nump = len(pgridreal)
    numxi = len(xigrid_real)
    numt = y_pred_transformed.shape[0] // (nump * numxi)
    InitgMax = 12.7417  # 6MeV
    InitgMin = 6.87084  # 3MeV
    # Reshape predictions
    unew = y_pred_transformed.detach().cpu().numpy().reshape(numxi, nump, numt)
    
    # Set up initial distribution based on case
    if case_name == 'isotropic':
        f_init_xi = np.array([0.5*(1-np.tanh((xi-2)/(0.25**10))) for xi in xigrid_real])
        f_p_RE_norm = f_p_RE / ((pgridreal * g_array  * 2))
    elif case_name == 'aligned':
        f_init_xi = np.array([0.5*(1-np.tanh((xi+0.9)/(0.25**10))) for xi in xigrid_real])
        f_p_RE_norm = f_p_RE / ((pgridreal * g_array * 0.1))
    else:  # opposed
        f_init_xi = np.array([0.5*(1+np.tanh((xi-0.9)/(0.25**10))) for xi in xigrid_real])
        f_p_RE_norm = f_p_RE / (pgridreal * g_array * 0.1)
    
    f_init_xi_norm = f_init_xi
    
    # Create initial distribution
    pgrid_reshape = pgridreal.reshape(1, nump, 1)
    f_init_ones = np.ones((numxi, nump))
    f_p_RE_norm_reshape = f_p_RE_norm.reshape(1, nump)
    f_init_xi_norm_reshape = f_init_xi_norm.reshape(numxi, 1)
    f_init = f_init_ones * f_p_RE_norm_reshape * f_init_xi_norm_reshape
    distribution = f_init.reshape(numxi, nump, 1)
    
    # Calculate density
    y = unew * distribution
    p_integrated = scipy.integrate.simpson(y * pgrid_reshape * pgrid_reshape, axis=1, x=pgrid_reshape)
    xi_integrated = scipy.integrate.simpson(p_integrated, axis=0, x=xigrid_real)
    
    return xi_integrated.flatten()
def plot_density_vs_time_pRE_variation(model, EF_val=2.5, Zeff_val=3.0, alpha_val=0.1, pRE_values_MeV=[1, 5, 10]):
    """Plot density vs time for different pRE values with fixed EF, Zeff, alpha and isotropic initial condition"""
    
    # Set up grid parameters (matching create_EF_variation_plot pattern)
    nump, numxi = 200, 201
    pgrid = np.linspace(0, 1, nump)
    pgridreal = np.linspace(pMin, pMax, nump)
    g_array = np.sqrt(pgridreal**2 + 1)
    InitgMax = 12.7417  # 6MeV
    InitgMin = 6.87084  # 3MeV
    xigrid = np.linspace(0, 1, numxi)
    xigrid_real = np.linspace(xiMin, xiMax, numxi)
    
    # Time grid (matching create_EF_variation_plot)
    numt = 100
    tgrid_real = np.linspace(0, 20, numt)
    tgrid = (tgrid_real - tMin) / (tMax - tMin)
    
    # Fixed parameters
    EF_norm = (EF_val - EFMin) / (EFMax - EFMin)
    Zeff_norm = (Zeff_val - ZeffMin) / (ZeffMax - ZeffMin)
    alpha_norm = (alpha_val - alphaMin) / (alphaMax - alphaMin)
    
    # Initial distribution setup (isotropic) - matching create_EF_variation_plot
    sigma_p = 3.5
    p_mid = np.sqrt((8/0.511 + 1)**2 - 1)
    f_p_RE = np.array([(np.sqrt(2*np.pi)*sigma_p)**-1 * np.exp(-0.5*((p-p_mid)/sigma_p)**2) for p in pgridreal])
    
    # Create figure (matching create_EF_variation_plot style)
    fig, ax = plt.subplots(1, 1)
    colors = ['r', 'g', 'b', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Process each pRE value
    for i, pRE_MeV in enumerate(pRE_values_MeV):
        # Convert pRE from MeV to normalized momentum (matching create_EF_variation_plot pattern)
        pRE_real = np.sqrt((1 + pRE_MeV*1e6/mecSQ)**2 - 1)
        pRE_norm = (pRE_real - pREMin) / (pREMax - pREMin)
        
        # Create meshgrid with single pRE value (matching create_EF_variation_plot pattern)
        pnew, xinew, pREnew, tnew, EFnew, Zeffnew, alphanew = np.meshgrid(
            pgrid, xigrid, [pRE_norm], tgrid,
            [EF_norm], [Zeff_norm], [alpha_norm]
        )
        
        X = np.vstack((np.ravel(pnew), np.ravel(xinew), np.ravel(pREnew),
                      np.ravel(tnew), np.ravel(EFnew), np.ravel(Zeffnew), np.ravel(alphanew))).T
        
        X_torch = torch.tensor(X, dtype=dtype, device=device)
        
        with torch.no_grad():
            y_pred = model(X_torch)
            y_pred_transformed = output_transform_7D(X_torch, y_pred)
        
        
        # Calculate density (matching create_EF_variation_plot - case_name defaults to 'isotropic')
        density = calculate_density(y_pred_transformed, pgridreal, g_array, 
                                   xigrid_real, f_p_RE, case_name='isotropic')
        
        # Clean up memory
        del X_torch, y_pred_transformed
        gc.collect()
        torch.cuda.empty_cache()
        
        # Plot density vs time (matching create_EF_variation_plot)
        ax.plot(tgrid_real, density, color=colors[i % len(colors)],
               label=f'$p_{{RE}} = {pRE_MeV}$ MeV', linewidth=3)
    
    ax.set_xlabel(r'$t/\tau_c$', fontsize=20)
    ax.set_ylabel(r'$n_{RE}/n_0$', fontsize=20)
    title_string = r"$E/E_c = $" + str(EF_val) + r", $Z_{eff}$ = " + str(Zeff_val) + r", $\alpha = $" + str(alpha_val)
    # ax.set_title(title_string, fontsize=20)
    ax.legend(loc='best', )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Save plot
    results_dir = f'{dataout}/results'
    os.makedirs(results_dir, exist_ok=True)
    filename = f'{results_dir}/density_vs_time_pRE_variation_EF={EF_val}_Zeff={Zeff_val}_alpha={alpha_val}_isotropic.png'
    fig.savefig(replace_dot_in_filename(filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Density vs time pRE variation plot saved")
def main():
    parser = argparse.ArgumentParser(description='PyTorch 7D PDE Solver')
    parser.add_argument('--model_path', type=str, default='pytorch_model_7D.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=1_000_000, help='Batch size for batched processing')
    parser.add_argument('--adaptive_sampling', action='store_true', help='Enable adaptive sampling (overrides --no_adaptive). Use: --adaptive_sampling --adaptive_freq 250')
    parser.add_argument('--adaptive_k', type=float, default=2.0, help='Power k for adaptive resampling')
    parser.add_argument('--adaptive_c', type=float, default=1.0, help='Constant c for adaptive resampling')
    parser.add_argument('--adaptive_freq', type=int, default=1000, help='Frequency of adaptive resampling (epochs)')
    # parser.add_argument('--adaptive_freq', type=int, default=2500, help='Frequency of adaptive resampling (epochs)')

    args = parser.parse_args()
    

    model = FCNN(7,architecture).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"7D Model loaded from {args.model_path}")
    
    print('Plotting the 7D energy distribution')
    plot_distribution_7D_gaussian(model, EF_val=2.5, Zeff_val=3.0, alpha_val=0.05)
    plot_distribution_7D_gaussian(model, EF_val=1.5, Zeff_val=1.0, alpha_val=0.2)
    plot_distribution_7D_gaussian(model, EF_val=4.0, Zeff_val=5.0, alpha_val=0.1)
    plot_density_vs_time_pRE_variation(model, EF_val=4.0, Zeff_val=3.0, alpha_val=0.1, pRE_values_MeV=[1, 5, 10])

    for EF_val in [2.5]:
        EF_norm = (EF_val - EFMin) / (EFMax - EFMin)
        for Zeff_val in [3.0]:
            Zeff_norm = (Zeff_val - ZeffMin) / (ZeffMax - ZeffMin)
            for alpha_val in [0.1]:
                alpha_norm = (alpha_val - alphaMin) / (alphaMax - alphaMin)
                plot_RPF_7D(model, pREgrid=0, tgrid=tMin, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=None)
                plot_RPF_7D(model, pREgrid=1, tgrid=tMin, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=None)
                plot_RPF_7D(model, pREgrid=0, tgrid=1.0, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=None)
                plot_RPF_7D(model, pREgrid=1, tgrid=1.0, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=None)

if __name__ == "__main__":
    main()