import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import os
import argparse
import scipy

dataout=os.path.dirname(os.path.abspath(__file__))+'/'
plt.rcParams.update({'font.size': 22})
# Set random seed for reproducibility
seed= 1234
torch.manual_seed(seed)
np.random.seed(seed)

# Configuration
dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Physical constants and parameters
mecSQ = 511e3  # electron rest mass in units eVs

# Domain parameters - 6D version (with variable Zeff and alpha)
EnergyMaxeV = 16e6
EnergyMineV = 1e4
gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)
xiMin, xiMax = -1, 1
tMin, tMax = 0, 20
# Electric field range
EFMin, EFMax = 1.5, 4.0  # Variable electric field range
# Zeff range
ZeffMin, ZeffMax = 1.0, 5.0  # Variable effective charge range
# Alpha range
alphaMin, alphaMax = 0.05, 0.2  # Variable radiation parameter range
gRE = 1+1e6/mecSQ
pRE_constant = np.sqrt(gRE**2-1)  # Fixed pRE value

print(f"pMax: {pMax}, pMin: {pMin}")
print(f"EF range: {EFMin} to {EFMax}")
print(f"Zeff range: {ZeffMin} to {ZeffMax}")
print(f"Alpha range: {alphaMin} to {alphaMax}")
print(f"pRE constant: {pRE_constant}")

# Training parameters
lr = 5.e-4
epochs = 50_000
num_domain_points = 5_000_000  # For 6D
num_test_points = 30_000_000   # For 6D
num_boundary_points = 500_000  # For 6D
T = 15e3
architecture= [64,64,64,64,64,64,1]


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
def output_transform_6D(inputs, outputs):
    """Output transform function for 6D problem"""
    pNorm, xiNorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6]
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    EF = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin
    Prob = torch.tanh(pNorm*outputs[:,0:1])
    return Prob

def plot_current_6D_comprehensive(model, EF_values, Zeff_values, alpha_values):
    """Create comprehensive current vs time plots varying over EF, Zeff, and alpha parameters"""
    
    # Set up common parameters
    nump, numxi = 200, 201
    pgrid = np.linspace(0, 1, nump)
    pgridreal = np.linspace(pMin, pMax, nump)
    g_array = np.sqrt(pgridreal**2 + 1)
    InitgMax = 12.7417  # 6MeV
    InitgMin = 6.87084  # 3MeV
    xigrid = np.linspace(0, 1, numxi)
    xigrid_real = np.linspace(xiMin, xiMax, numxi)
    numt = 20
    tgrid_real = np.linspace(0, 20, numt)
    tgrid = (tgrid_real - tMin) / (tMax - tMin)
    
    # Normalize parameter arrays
    EFgrid = (np.array(EF_values) - EFMin) / (EFMax - EFMin)
    Zeffgrid = (np.array(Zeff_values) - ZeffMin) / (ZeffMax - ZeffMin)
    alphagrid = (np.array(alpha_values) - alphaMin) / (alphaMax - alphaMin)
    
    # Set up initial distribution
    pMin_init = np.sqrt(InitgMin**2 - 1)
    pMax_init = np.sqrt(InitgMax**2 - 1)
    sigma_p = 3.5
    p_mid = np.sqrt((8/0.511 + 1)**2 - 1)
    f_p_RE = np.array([(np.sqrt(2*np.pi)*sigma_p)**-1 * np.exp(-0.5*((p-p_mid)/sigma_p)**2) for p in pgridreal])
    
    # Color lists for different parameters
    colors = ['r', 'g', 'b', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create three types of plots
    create_EF_variation_plot(model, EF_values, Zeff_values, alpha_values, 
                            pgrid, pgridreal, g_array, xigrid, xigrid_real, 
                            tgrid, tgrid_real, f_p_RE, colors)
    
    create_Zeff_variation_plot(model, EF_values, Zeff_values, alpha_values,
                              pgrid, pgridreal, g_array, xigrid, xigrid_real,
                              tgrid, tgrid_real, f_p_RE, colors)
    
    create_alpha_variation_plot(model, EF_values, Zeff_values, alpha_values,
                               pgrid, pgridreal, g_array, xigrid, xigrid_real,
                               tgrid, tgrid_real, f_p_RE, colors)

def create_EF_variation_plot(model, EF_values, Zeff_values, alpha_values,
                            pgrid, pgridreal, g_array, xigrid, xigrid_real,
                            tgrid, tgrid_real, f_p_RE, colors):
    """Plot current vs time for varying EF values, fixed Zeff and alpha"""
    # Use middle values for fixed parameters
    fixed_Zeff = Zeff_values[len(Zeff_values)//2]
    fixed_alpha = alpha_values[len(alpha_values)//2]
    
    fig, ax = plt.subplots(1, 1)
    
    for i, EF_val in enumerate(EF_values):
        # Create meshgrid for this EF value
        pnew, xinew, tnew, EFnew, Zeffnew, alphanew = np.meshgrid(
            pgrid, xigrid, tgrid, 
            [(EF_val - EFMin) / (EFMax - EFMin)], 
            [(fixed_Zeff - ZeffMin) / (ZeffMax - ZeffMin)],
            [(fixed_alpha - alphaMin) / (alphaMax - alphaMin)]
        )
        
        X = np.vstack((np.ravel(pnew), np.ravel(xinew), np.ravel(tnew), 
                      np.ravel(EFnew), np.ravel(Zeffnew), np.ravel(alphanew))).T
        
        X_torch = torch.tensor(X, dtype=dtype, device=device)
        
        with torch.no_grad():
            y_pred = model(X_torch)
            y_pred_transformed = output_transform_6D(X_torch, y_pred)
        
        # Calculate current
        case = 'isotropic'
        current = calculate_current(y_pred_transformed, pgridreal, g_array, 
                                  xigrid_real, f_p_RE,case)
        
        ax.plot(tgrid_real, current, color=colors[i % len(colors)], 
               label=f'$E_\Vert = {EF_val:.2f}$', linewidth=1)
        
        # Add jonta particle data (isotropic orientation for EF variation)
        jonta_t, jonta_j = load_jonta_data(EF_val, fixed_Zeff, fixed_alpha, orientation=case)
        if jonta_t is not None:
            ax.plot(jonta_t, jonta_j, color=colors[i % len(colors)], 
                   linestyle='--', alpha=0.7, linewidth=1.0)
        jonta_t, jonta_j = load_jonta_data_full(EF_val, fixed_Zeff, fixed_alpha, orientation=case)
        if jonta_t is not None:
            ax.scatter(jonta_t, jonta_j, color=colors[i % len(colors)], marker= 'x',
                        alpha=0.7, linewidth=1.0)
    
    ax.set_xlabel(r'$t/\tau_c$')
    ax.set_ylabel(r'$\frac{u_\Vert n_{RE}}{c n_{RE}(0)}$', fontsize=22)
    ax.set_title(f'$Z_{{eff}}={fixed_Zeff}$, $\\alpha={fixed_alpha}$', fontsize=22)
    ax.legend(loc='best', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Save plot
    results_dir = f'{dataout}/results'
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(f'{results_dir}/current_vs_time_EF_variation_{case}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"EF variation plot saved")

def create_Zeff_variation_plot(model, EF_values, Zeff_values, alpha_values,
                              pgrid, pgridreal, g_array, xigrid, xigrid_real,
                              tgrid, tgrid_real, f_p_RE, colors):
    """Plot current vs time for varying Zeff values, fixed EF and alpha"""
    # Use middle values for fixed parameters
    fixed_EF = EF_values[len(EF_values)//2]
    fixed_alpha = alpha_values[len(alpha_values)//2]
    
    fig, ax = plt.subplots(1, 1)
    
    for i, Zeff_val in enumerate(Zeff_values):
        # Create meshgrid for this Zeff value
        pnew, xinew, tnew, EFnew, Zeffnew, alphanew = np.meshgrid(
            pgrid, xigrid, tgrid,
            [(fixed_EF - EFMin) / (EFMax - EFMin)],
            [(Zeff_val - ZeffMin) / (ZeffMax - ZeffMin)],
            [(fixed_alpha - alphaMin) / (alphaMax - alphaMin)]
        )
        
        X = np.vstack((np.ravel(pnew), np.ravel(xinew), np.ravel(tnew),
                      np.ravel(EFnew), np.ravel(Zeffnew), np.ravel(alphanew))).T
        
        X_torch = torch.tensor(X, dtype=dtype, device=device)
        
        with torch.no_grad():
            y_pred = model(X_torch)
            y_pred_transformed = output_transform_6D(X_torch, y_pred)
        
        # Calculate current
        case = 'isotropic'
        current = calculate_current(y_pred_transformed, pgridreal, g_array,
                                  xigrid_real, f_p_RE,case)
        
        ax.plot(tgrid_real, current, color=colors[i % len(colors)],
               label=f'$Z_{{eff}} = {Zeff_val:.1f}$', linewidth=1)
        
        # Add jonta particle data (aligned orientation for Zeff variation)
        jonta_t, jonta_j = load_jonta_data(fixed_EF, Zeff_val, fixed_alpha, orientation=case)
        if jonta_t is not None:
            ax.plot(jonta_t, jonta_j, color=colors[i % len(colors)],
                   linestyle='--', alpha=0.7, linewidth=1.0)
        jonta_t, jonta_j = load_jonta_data_full(fixed_EF, Zeff_val, fixed_alpha, orientation=case)
        if jonta_t is not None:
            ax.scatter(jonta_t, jonta_j, color=colors[i % len(colors)],  marker= 'x',
                        alpha=0.7, linewidth=1.0)
    
    ax.set_xlabel(r'$t/\tau_c$', fontsize=22)
    ax.set_ylabel(r'$\frac{u_\Vert n_{RE}}{c n_{RE}(0)}$', fontsize=22)
    ax.set_title(f'$E_\Vert={fixed_EF}$, $\\alpha={fixed_alpha}$', fontsize=22)
    ax.legend(loc='best', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Save plot
    results_dir = f'{dataout}/results'
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(f'{results_dir}/current_vs_time_Zeff_variation_{case}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Zeff variation plot saved")

def create_alpha_variation_plot(model, EF_values, Zeff_values, alpha_values,
                               pgrid, pgridreal, g_array, xigrid, xigrid_real,
                               tgrid, tgrid_real, f_p_RE, colors):
    """Plot current vs time for varying alpha values, fixed EF and Zeff"""
    # Use middle values for fixed parameters
    fixed_EF = EF_values[len(EF_values)//2]
    fixed_Zeff = Zeff_values[len(Zeff_values)//2]
    
    fig, ax = plt.subplots(1, 1)
    
    for i, alpha_val in enumerate(alpha_values):
        # Create meshgrid for this alpha value
        pnew, xinew, tnew, EFnew, Zeffnew, alphanew = np.meshgrid(
            pgrid, xigrid, tgrid,
            [(fixed_EF - EFMin) / (EFMax - EFMin)],
            [(fixed_Zeff - ZeffMin) / (ZeffMax - ZeffMin)],
            [(alpha_val - alphaMin) / (alphaMax - alphaMin)]
        )
        
        X = np.vstack((np.ravel(pnew), np.ravel(xinew), np.ravel(tnew),
                      np.ravel(EFnew), np.ravel(Zeffnew), np.ravel(alphanew))).T
        
        X_torch = torch.tensor(X, dtype=dtype, device=device)
        
        with torch.no_grad():
            y_pred = model(X_torch)
            y_pred_transformed = output_transform_6D(X_torch, y_pred)
        
        # Calculate current
        case = 'isotropic'
        current = calculate_current(y_pred_transformed, pgridreal, g_array,
                                  xigrid_real, f_p_RE,case)
        
        ax.plot(tgrid_real, current, color=colors[i % len(colors)],
               label=f'α = {alpha_val:.3f}', linewidth=1.0)
        
        # Add jonta particle data (opposed orientation for alpha variation)
        jonta_t, jonta_j = load_jonta_data(fixed_EF, fixed_Zeff, alpha_val, orientation=case)
        if jonta_t is not None:
            ax.plot(jonta_t, jonta_j, color=colors[i % len(colors)],
                   linestyle='--', alpha=0.7, linewidth=1.0)
        jonta_t, jonta_j = load_jonta_data_full(fixed_EF, fixed_Zeff, alpha_val, orientation=case)
        if jonta_t is not None:
            ax.scatter(jonta_t, jonta_j, color=colors[i % len(colors)],  marker= 'x',
                        alpha=0.7, linewidth=1.0)

    ax.set_xlabel(r'$t/\tau_c$', fontsize=22)
    ax.set_ylabel(r'$\frac{u_\Vert n_{RE}}{c n_{RE}(0)}$', fontsize=22)
    ax.set_title(f'$E_\Vert={fixed_EF}$, $Z_{{eff}}={fixed_Zeff}$', fontsize=22)
    ax.legend(loc='best', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Save plot
    results_dir = f'{dataout}/results'
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(f'{results_dir}/current_vs_time_alpha_variation_{case}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Alpha variation plot saved")

def load_jonta_data(EF_val, Zeff_val, alpha_val, orientation='isotropic'):
    """Load particle data from jonta folder for given parameters with specified orientation
    
    Args:
        EF_val: Electric field value
        Zeff_val: Effective charge value
        alpha_val: Alpha parameter value
        orientation: Orientation type - 'isotropic', 'aligned', or 'opposed'
    """
    try:
        jonta_dir = f"REmoments_E={"%.2f"%EF_val}_Zeff={"%.2f"%Zeff_val}_alpha={"%.4f"%alpha_val}_{orientation}/data/j_vs_time.txt"
        X, y = np.loadtxt(jonta_dir).T
        return X/1000, y  # Convert time units
    except FileNotFoundError:
        print(f"Warning: Jonta data not found for EF={EF_val}, Zeff={Zeff_val}, alpha={alpha_val}, orientation={orientation}")
        return None, None
def load_jonta_data_full(EF_val, Zeff_val, alpha_val, orientation='isotropic'):
    """Load particle data from jonta folder for given parameters with specified orientation (full version)
    
    Args:
        EF_val: Electric field value
        Zeff_val: Effective charge value
        alpha_val: Alpha parameter value
        orientation: Orientation type - 'isotropic', 'aligned', or 'opposed'
    """
    try:
        jonta_dir = f"REmoments_E={"%.2f"%EF_val}_Zeff={"%.2f"%Zeff_val}_alpha={"%.4f"%alpha_val}_{orientation}_full/data/j_vs_time.txt"
        X, y = np.loadtxt(jonta_dir).T
        return X/1000, y  # Convert time units
    except FileNotFoundError:
        print(f"Warning: Jonta data not found for EF={EF_val}, Zeff={Zeff_val}, alpha={alpha_val}, orientation={orientation}")
        return None, None
def calculate_current(y_pred_transformed, pgridreal, g_array, xigrid_real, f_p_RE,case):
    """Calculate current by integrating over momentum and pitch angle"""
    nump = len(pgridreal)
    numxi = len(xigrid_real)
    numt = y_pred_transformed.shape[0] // (nump * numxi)
    
    # Reshape predictions
    unew = y_pred_transformed.detach().cpu().numpy().reshape(numxi, nump, numt)
    
    # Set up initial distribution (aligned case from xi=-1 to -0.9)
    # f_init_xi = np.array([0.5*(1-np.tanh((xi+0.9)/(0.25**10))) for xi in xigrid_real])
    # Set up initial distribution (isotropic case)
    if case == 'isotropic':
        f_init_xi = np.array([0.5*(1-np.tanh((xi-2)/(0.25**10))) for xi in xigrid_real])
        f_p_RE_norm = f_p_RE / ((pgridreal * g_array * 2))
    elif case == 'aligned':
        f_init_xi = np.array([0.5*(1-np.tanh((xi+0.9)/(0.25**10))) for xi in xigrid_real])
        f_p_RE_norm = f_p_RE / ((pgridreal * g_array*0.1))
    elif case == 'opposed':
        f_init_xi = np.array([0.5*(1+np.tanh((xi-0.9)/(0.25**10))) for xi in xigrid_real])
        f_p_RE_norm = f_p_RE / ((pgridreal * g_array*0.1))
    f_init_xi_norm = f_init_xi
    
    # Create initial distribution
    pgrid_reshape = pgridreal.reshape(1, nump, 1)
    f_init_ones = np.ones((numxi, nump))
    f_p_RE_norm_reshape = f_p_RE_norm.reshape(1, nump)
    f_init_xi_norm_reshape = f_init_xi_norm.reshape(numxi, 1)
    f_init = f_init_ones * f_p_RE_norm_reshape * f_init_xi_norm_reshape
    distribution = f_init.reshape(numxi, nump, 1)
    
    # Calculate current
    y = unew * distribution
    p_integrated = scipy.integrate.simpson(y * pgrid_reshape * pgrid_reshape, axis=1, x=pgrid_reshape)
    xi_integrated = scipy.integrate.simpson(p_integrated, axis=0, x=xigrid_real)
    
    return xi_integrated.flatten()

def plot_orientation_comparison(model, EF_val, Zeff_val, alpha_val,
                               pgrid, pgridreal, g_array, xigrid, xigrid_real,
                               tgrid, tgrid_real, f_p_RE):
    """Plot current vs time for different orientations at fixed E, Zeff, alpha"""
    
    fig, ax = plt.subplots(1, 1)
    
    orientations = ['aligned', 'isotropic', 'opposed']
    colors = ['r', 'g', 'b']
    labels = [r'$\xi_0 \in (-1.0,-0.9)$', 
              r'$\xi_0 \in (-1,1)$', 
              r'$\xi_0 \in (0.9,1.0)$']
    
    for i, case in enumerate(orientations):
        # Create meshgrid for this orientation
        pnew, xinew, tnew, EFnew, Zeffnew, alphanew = np.meshgrid(
            pgrid, xigrid, tgrid,
            [(EF_val - EFMin) / (EFMax - EFMin)],
            [(Zeff_val - ZeffMin) / (ZeffMax - ZeffMin)],
            [(alpha_val - alphaMin) / (alphaMax - alphaMin)]
        )
        
        X = np.vstack((np.ravel(pnew), np.ravel(xinew), np.ravel(tnew),
                      np.ravel(EFnew), np.ravel(Zeffnew), np.ravel(alphanew))).T
        
        X_torch = torch.tensor(X, dtype=dtype, device=device)
        
        with torch.no_grad():
            y_pred = model(X_torch)
            y_pred_transformed = output_transform_6D(X_torch, y_pred)
        
        # Calculate current
        current = calculate_current(y_pred_transformed, pgridreal, g_array,
                                  xigrid_real, f_p_RE, case)
        
        ax.plot(tgrid_real, current, color=colors[i], 
               label=labels[i], linewidth=1.5)
        
        # Add jonta particle data with dashed lines
        jonta_t, jonta_j = load_jonta_data(EF_val, Zeff_val, alpha_val, orientation=case)
        if jonta_t is not None:
            ax.plot(jonta_t, jonta_j, color=colors[i], 
                   linestyle='--', alpha=0.7, linewidth=1.0)
        jonta_t, jonta_j = load_jonta_data_full(EF_val, Zeff_val, alpha_val, orientation=case)
        if jonta_t is not None:
            ax.scatter(jonta_t, jonta_j, color=colors[i], marker='x',
                        alpha=0.7, linewidth=1.0)
    
    ax.set_xlabel(r'$t/\tau_c$', fontsize=22)
    ax.set_ylabel(r'$\frac{u_\Vert n_{RE}}{c n_{RE}(0)}$', fontsize=22)
    title_str = f'$E_\Vert={EF_val}$, $Z_{{eff}}={Zeff_val}$, $\\alpha={alpha_val}$'
    ax.set_title(title_str, fontsize=22)
    ax.legend(loc='best', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Save plot
    results_dir = f'{dataout}/results'
    os.makedirs(results_dir, exist_ok=True)
    EF_str = str(EF_val).replace('.', 'o')
    Zeff_str = str(Zeff_val).replace('.', 'o')
    alpha_str = str(alpha_val).replace('.', 'o')
    fig.savefig(f'{results_dir}/current_vs_time_orientation_comparison_E{EF_str}_Zeff{Zeff_str}_alpha{alpha_str}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Orientation comparison plot saved")

def load_model(model_path):
    """Load a trained model from file"""
    model = FCNN(6,architecture).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model



def main():
    parser = argparse.ArgumentParser(description='PyTorch 6D PDE Solver')
    parser.add_argument('--model_path', type=str, default='pytorch_model_6D.pth', help='Path to trained model')
     
    args = parser.parse_args()
    
    model = FCNN(6,architecture).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"6D Model loaded from {args.model_path}")
    
    # Create comprehensive parameter variation plots
    print('Creating comprehensive parameter variation plots...')
    EF_values = [4.0,2.5,1.5]
    Zeff_values = [1.0, 3.0, 5.0]
    alpha_values = [0.05,0.1,0.2]
    
    plot_current_6D_comprehensive(model, EF_values, Zeff_values, alpha_values)
    # Create orientation comparison plot for E=2.5, Zeff=3.0, alpha=0.1
    # print('Creating orientation comparison plot...')
    # # Set up common parameters (same as in comprehensive plots)
    nump, numxi = 200, 201
    pgrid = np.linspace(0, 1, nump)
    pgridreal = np.linspace(pMin, pMax, nump)
    g_array = np.sqrt(pgridreal**2 + 1)
    xigrid = np.linspace(0, 1, numxi)
    xigrid_real = np.linspace(xiMin, xiMax, numxi)
    numt = 20
    tgrid_real = np.linspace(0, 20, numt)
    tgrid = (tgrid_real - tMin) / (tMax - tMin)
    
    # # Set up initial distribution
    sigma_p = 3.5
    p_mid = np.sqrt((8/0.511 + 1)**2 - 1)
    f_p_RE = np.array([(np.sqrt(2*np.pi)*sigma_p)**-1 * np.exp(-0.5*((p-p_mid)/sigma_p)**2) for p in pgridreal])
    
    plot_orientation_comparison(model, EF_val=2.5, Zeff_val=3.0, alpha_val=0.1,
                                pgrid=pgrid, pgridreal=pgridreal, g_array=g_array,
                                xigrid=xigrid, xigrid_real=xigrid_real,
                                tgrid=tgrid, tgrid_real=tgrid_real, f_p_RE=f_p_RE)

if __name__ == "__main__":
    main()

