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

def tMin_func_7D(X):
    """Initial condition function for 7D"""
    pNorm, xiNorm, pRENorm, tNorm, EFNorm, ZeffNorm, alphaNorm = X.split(1, dim=1)
    p = (pMax-pMin)*pNorm+pMin
    pRE = (pREMax-pREMin)*pRENorm+pREMin
    pRENorm = (pRE-pREMin)/(pREMax-pREMin)

    N_0 = 100
    N_1 = 30
    delta = N_1 / (N_0 - N_1)
    N_init = N_0 * delta
    N = (N_init / (delta + pRENorm**2))
    # N = 64
    Prob = (pRE - p) / (pMax - pMin) * N
    Pinit = 0.5 * (1.0 - torch.tanh(Prob))
    return Pinit

def pMax_func_7D(inputs):
    """Function that defines what the solution should be at pMax boundary - 7D version"""
    pNorm, xiNorm, pRENorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs.split(1, dim=1)
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    EFval = (EFMax-EFMin)*EFNorm+EFMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin

    at_pMax = torch.isclose(p, torch.tensor(pMax, dtype=dtype, device=device), atol=1e-6)
    
    Up = -EFval * xi - (1 + p**2) / (p**2) - alpha * p * torch.sqrt(1 + p**2) * (1 - xi**2)
    runaway_mask = Up > 0
    
    combined_mask = at_pMax & runaway_mask
    
    solution_value = torch.ones_like(combined_mask, dtype=dtype, device=device)
    
    return torch.where(combined_mask, solution_value, torch.zeros_like(combined_mask, dtype=dtype, device=device))

def pMax_bound_7D(inputs):
    """Boundary condition for pMax - 7D version"""
    pNorm, xiNorm, pRENorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs.split(1, dim=1)
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    EFval = (EFMax-EFMin)*EFNorm+EFMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin

    at_pMax = torch.isclose(p, torch.tensor(pMax, dtype=dtype, device=device), atol=1e-6)
    
    Up = -EFval * xi - (1 + p**2) / (p**2) - alpha * p * torch.sqrt(1 + p**2) * (1 - xi**2)
    runaway_mask = Up > 0
    
    combined_mask = at_pMax & runaway_mask
    return combined_mask.squeeze()

def tMin_bound_7D(inputs):
    """Boundary condition for tMin - 7D version"""
    pNorm, xiNorm, pRENorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs.split(1, dim=1)
    t = (tMax-tMin)*tNorm+tMin
    mask = torch.isclose(t, torch.tensor(tMin, dtype=dtype, device=device), atol=1e-6)
    return mask.squeeze()
def generate_training_points_7D(num_domain_points):
    """Generate training points for 7D domain and boundary - following 6D sampling pattern"""
    engine = SobolEngine(dimension=7, scramble=True, seed=seed)
    domain_points = engine.draw(num_domain_points,dtype=dtype).to(device).requires_grad_(True)
    
    # Add boundary points directly to domain_points (similar to 6D script)
    # Generate boundary points and add to domain points
    num_boundary_per_type = num_domain_points // 20  # 5% of domain points per boundary type
    
    # xi=-1 boundary points (dimension 1 = 0 in normalized coordinates)
    xi_minus1_points = engine.draw(num_boundary_per_type, dtype=dtype).to(device)
    xi_minus1_points[:, 1] = 0  # Set xi to xiMin (-1) in normalized coordinates
    xi_minus1_points = xi_minus1_points.requires_grad_(True)
    
    # xi=+1 boundary points (dimension 1 = 1 in normalized coordinates)
    xi_plus1_points = engine.draw(num_boundary_per_type, dtype=dtype).to(device)
    xi_plus1_points[:, 1] = 1  # Set xi to xiMax (+1) in normalized coordinates
    xi_plus1_points = xi_plus1_points.requires_grad_(True)
    
    # Add boundary points to domain points
    domain_points = torch.cat([domain_points, xi_minus1_points, xi_plus1_points], dim=0).requires_grad_(True)
    
    total_boundary_added = 2 * num_boundary_per_type
    print(f"Added {total_boundary_added} boundary points to domain:")
    print(f"  - {num_boundary_per_type} xi=-1 points")
    print(f"  - {num_boundary_per_type} xi=+1 points")
    print(f"Total domain points: {len(domain_points)}")

    pMax_points = engine.draw(num_boundary_points // 2,dtype=dtype)
    pMax_points[:, 0] = 1
    
    # Convert to physical coordinates
    alpha_norm_pMax = pMax_points[:, 6]
    alpha_pMax = alpha_norm_pMax * (alphaMax - alphaMin) + alphaMin
    E_norm_pMax = pMax_points[:, 4]
    E_pMax = E_norm_pMax * (EFMax - EFMin) + EFMin
    
    # Solve quadratic equation: A * xi² + B * xi + C = 0
    # Where Up = -E * xi - (1 + p²)/p² - alpha * p * sqrt(1 + p²) * (1 - xi²) = 0
    # Rearranged: alpha * p * sqrt(1 + p²) * xi² - E * xi - (1 + p²)/p² - alpha * p * sqrt(1 + p²) = 0
    A = alpha_pMax * np.sqrt(pMax**2 + 1) * pMax
    B = -E_pMax
    C = -(1 + pMax**2) / pMax**2 - alpha_pMax * np.sqrt(pMax**2 + 1) * pMax
    
    # Calculate discriminant
    discriminant = B**2 - 4 * A * C
    
    # Calculate both roots
    xiMax1 = (-B + np.sqrt(np.maximum(discriminant, 0))) / (2 * A)
    xiMax2 = (-B - np.sqrt(np.maximum(discriminant, 0))) / (2 * A)
    
    # Choose the valid root (between -1 and 1)
    valid1 = (xiMax1 >= -1) & (xiMax1 <= 1) & (discriminant >= 0)
    valid2 = (xiMax2 >= -1) & (xiMax2 <= 1) & (discriminant >= 0)
    
    # Select the appropriate root for each point
    xiMax_pMax = np.where(valid1, xiMax1, xiMax2)
    xiMax_pMax = np.where(valid2 & ~valid1, xiMax2, xiMax_pMax)
    
    # Handle cases where no valid solution exists (fallback to xi = -1)
    no_valid_solution = ~(valid1 | valid2)
    xiMax_pMax = np.where(no_valid_solution, -1.0, xiMax_pMax)
    
    # Sample xi in the range [-1, xiMax_pMax] where Up > 0
    xiMax_norm_pMax = (xiMax_pMax - xiMin) / (xiMax - xiMin)
    
    # Set the new xi values
    pMax_points[:, 1] = pMax_points[:, 1]  * xiMax_norm_pMax
    pMax_points= pMax_points.to(device).requires_grad_(True)
    
    tMin_points = engine.draw(num_boundary_points//2, dtype=dtype).to(device)
    tMin_points[:, 3] = 0
    tMin_points = tMin_points.requires_grad_(True)
    
    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    return domain_points, tMin_points, pMax_points
def adaptive_resample_points(model, domain_points, k=2, c=1.0, num_points_to_add=None, batch_size=10000):
    """Adaptive point resampling based on PDE residual with batched processing"""
    if num_points_to_add is None:
        num_points_to_add = len(domain_points) // 4
    
    print(f"Adaptive resampling: computing residuals for {len(domain_points)} points in batches...")
    
    # Compute PDE residuals for current domain points in batches
    all_residuals = []
    num_batches = (len(domain_points) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(domain_points))
        batch_points = domain_points[start_idx:end_idx].detach().requires_grad_(True)
        
        y_pred = model(batch_points)
        y_pred_transformed = output_transform_7D(batch_points, y_pred)
        pde_residuals = pde_residual_7D(batch_points, y_pred_transformed)
        
        batch_residuals = torch.abs(pde_residuals).detach().cpu().numpy()
        all_residuals.append(batch_residuals)

        # Aggressive memory cleanup
        del batch_points, y_pred, y_pred_transformed, pde_residuals, batch_residuals
        gc.collect()
        torch.cuda.empty_cache()
    
    residuals_abs = np.concatenate(all_residuals)
    
    # Calculate probability distribution
    # if np.power(residuals_abs, k).max() != 0:
    resid_mean=np.abs(residuals_abs)/np.abs(residuals_abs).mean()
    err_eq_domain = np.power(resid_mean, k) + c
    err_eq_flatten = err_eq_domain.flatten()
    err_eq_normalized = err_eq_flatten / err_eq_flatten.sum()
    
    # Sample new points based on probability distribution
    points_ids = np.random.choice(
        a=len(err_eq_normalized), 
        size=num_points_to_add, 
        replace=False, 
        p=err_eq_normalized
    )
    
    new_domain_points = domain_points[points_ids].detach().requires_grad_(True)
    # Print statistics about the resampling
    print(f"Adaptive resampling completed: replaced {num_points_to_add} points")
    print(f"Residual statistics: mean={(residuals_abs**2).mean():.6e}, max={residuals_abs.max():.6e}, std={residuals_abs.std():.6e}")
    
    # Clean up memory
    del all_residuals, residuals_abs, err_eq_domain, err_eq_flatten, err_eq_normalized
    gc.collect()
    torch.cuda.empty_cache()

    return new_domain_points
def compute_losses_7D(model, domain_points_total, tMin_points, pMax_points):
    """Compute all loss components for 7D problem"""
    domain_points_total.requires_grad_(True)
    domain_outputs = model(domain_points_total)
    domain_outputs_transformed = output_transform_7D(domain_points_total, domain_outputs)
    resid = pde_residual_7D(domain_points_total, domain_outputs_transformed)
    
    model_tMin = model(tMin_points)
    output_tMin = output_transform_7D(tMin_points, model_tMin)
    tMin_target = tMin_func_7D(tMin_points)
    tMin_loss = output_tMin - tMin_target
    
    model_pMax = model(pMax_points)
    output_pMax = output_transform_7D(pMax_points,model_pMax)
    pMax_target = pMax_func_7D(pMax_points)
    pMax_loss = output_pMax - pMax_target

    return resid, pMax_loss, tMin_loss

def compute_losses_batched_7D(model, domain_points_total, tMin_points, pMax_points, batch_size=10000):
    """Compute all loss components with batched processing for 7D"""
    total_pde_loss = 0.0
    num_batches = (len(domain_points_total) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(domain_points_total))
        batch_points = domain_points_total[start_idx:end_idx].detach().requires_grad_(True)
        
        batch_outputs = model(batch_points)
        batch_outputs_transformed = output_transform_7D(batch_points, batch_outputs)
        batch_resid = pde_residual_7D(batch_points, batch_outputs_transformed)
        

        batch_pde_loss = torch.mean(batch_resid**2)
        
        total_pde_loss += batch_pde_loss.item() * (end_idx - start_idx)

        # Aggressive memory cleanup
        del batch_points, batch_outputs, batch_outputs_transformed, batch_resid
        if 'batch_weights' in locals(): del batch_weights
        if 'weighted_batch_resid' in locals(): del weighted_batch_resid
        gc.collect()
        torch.cuda.empty_cache()
    
    pde_loss = torch.tensor(total_pde_loss / len(domain_points_total), dtype=dtype, device=device)
    
    model_tMin = model(tMin_points)
    outputs_tMin = output_transform_7D(tMin_points,model_tMin)
    tMin_target = tMin_func_7D(tMin_points)
    tMin_loss = torch.mean((outputs_tMin-tMin_target)**2)

    model_pMax = model(pMax_points)
    outputs_pMax = output_transform_7D(pMax_points,model_pMax)
    pMax_target = pMax_func_7D(pMax_points)
    pMax_loss = torch.mean((outputs_pMax - pMax_target)**2)
    

    del domain_points_total,pMax_points, tMin_points, outputs_pMax, outputs_tMin
    gc.collect()
    torch.cuda.empty_cache()
    total_loss = pde_loss + pMax_loss + tMin_loss
    
    return total_loss, pde_loss, pMax_loss, tMin_loss
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

def train_model_7D(args):
    """Main training function for 7D problem"""
    model = FCNN(7,architecture).to(device)
    # model.load_state_dict(torch.load(args.model_path, map_location=device))
    # print(f"7D Model loaded from {args.model_path}")
    optimizer = Soap(model.parameters(), lr=lr,betas=(0.997,0.997))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999995)
    
    domain_points, tMin_points, pMax_points = generate_training_points_7D(num_domain_points)
    domain_points_total = domain_points
    test_points, tMin_points_test, pMax_points_test = generate_training_points_7D(num_test_points)
    test_points_total = test_points
    test_points_total.detach().cpu()
    
    pbar = tqdm.trange(epochs)

    total_test_loss_str ="--------"
    pMax_test_loss_str = "--------"
    tMin_test_loss_str = "--------"

    soap_train_loss_history = []
    soap_train_pde_loss_history = []
    soap_train_bc_loss_history = []
    soap_train_ic_loss_history = []
    soap_test_loss_history = []
    soap_test_pde_loss_history = []
    soap_test_bc_loss_history = []
    soap_test_ic_loss_history = []
    soap_epoch_history = []

    soap_header = "Epoch,Train_Total_Loss,Train_PDE_Loss,Train_BC_Loss,Train_IC_Loss,Test_Total_Loss,Test_PDE_Loss,Test_BC_Loss,Test_IC_Loss"
    
    data_dir = f'{dataout}/data'
    os.makedirs(data_dir, exist_ok=True)
    for epoch in pbar:
        optimizer.zero_grad()
        
        domain_points_total = domain_points_total.detach().requires_grad_(True)
        tMin_points = tMin_points.detach().requires_grad_(True)
        pMax_points = pMax_points.detach().requires_grad_(True)
        pde_resid, pMax_resid,tMin_resid= compute_losses_7D(model, domain_points_total, tMin_points,pMax_points)
        
        pde_loss = torch.mean(pde_resid**2)
        pMax_loss = torch.mean(pMax_resid**2)
        tMin_loss = torch.mean(tMin_resid**2)
            
        total_loss = pde_loss + pMax_loss +  tMin_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        soap_train_loss_history.append(total_loss.item())
        soap_train_pde_loss_history.append(pde_loss.item())
        soap_train_bc_loss_history.append(pMax_loss.item())
        soap_train_ic_loss_history.append(tMin_loss.item())
        
        pbar.set_postfix({
            "Train PDE": f"{pde_loss.item():.2e}",
            "Train IC": f"{tMin_loss.item():.2e}",
            "Train BC": f"{pMax_loss.item():.2e}",
            "Test PDE": f"{total_test_loss_str}",
            "Test IC": f"{tMin_test_loss_str}",
            "Test BC": f"{pMax_test_loss_str}",
        })

        soap_epoch_history.append(epoch)
        soap_test_loss_history.append(0) 
        soap_test_pde_loss_history.append(0)
        soap_test_bc_loss_history.append(0)
        soap_test_ic_loss_history.append(0)

        if (epoch % 5000 == 0) & (epoch!=0) :
            domain_points_total.detach().cpu()
            test_points_total.to(device)
            total_test_loss, pde_test_loss, pMax_test_loss, tMin_test_loss = compute_losses_batched_7D(model, test_points_total, tMin_points_test, pMax_points_test,batch_size=args.batch_size)
            test_points.detach().cpu()
            domain_points_total.to(device)
            total_test_loss_str = f"{pde_test_loss.item():.2e}"
            pMax_test_loss_str = f"{pMax_test_loss.item():.2e}" 
            tMin_test_loss_str = f"{tMin_test_loss.item():.2e}" 

            soap_test_loss_history[-1] = total_test_loss.item()
            soap_test_pde_loss_history[-1] = pde_test_loss.item()
            soap_test_bc_loss_history[-1] = pMax_test_loss.item()
            soap_test_ic_loss_history[-1] = tMin_test_loss.item()

            soap_loss_data = np.column_stack([
                soap_epoch_history,
                soap_train_loss_history,
                soap_train_pde_loss_history,
                soap_train_bc_loss_history,
                soap_train_ic_loss_history,
                soap_test_loss_history,
                soap_test_pde_loss_history,
                soap_test_bc_loss_history,
                soap_test_ic_loss_history
            ])
            
            np.savetxt(f'{data_dir}/soap_loss_history_7d.csv', soap_loss_data, delimiter=',', header=soap_header, comments='')
            for EF_val in [1.5, 2.0, 2.5]:
                EF_norm = (EF_val - EFMin) / (EFMax - EFMin)
                for Zeff_val in [1.0, 3.0, 5.0]:
                    Zeff_norm = (Zeff_val - ZeffMin) / (ZeffMax - ZeffMin)
                    for alpha_val in [0.05, 0.1,0.2]:
                        alpha_norm = (alpha_val - alphaMin) / (alphaMax - alphaMin)
                        plot_RPF_7D(model, pREgrid=0, tgrid=tMin, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=domain_points)
                        plot_RPF_7D(model, pREgrid=1, tgrid=tMin, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=domain_points)
                        plot_RPF_7D(model, pREgrid=0, tgrid=0.5, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=domain_points)
                        plot_RPF_7D(model, pREgrid=1, tgrid=0.5, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=domain_points)
                        plot_RPF_7D(model, pREgrid=0, tgrid=1, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=domain_points)
                        plot_RPF_7D(model, pREgrid=1, tgrid=1, EFgrid=EF_norm, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, losses=None, domain_points=domain_points)
                
            torch.save(model.state_dict(), f'{dataout}/pytorch_model_7D.pth')
        if (epoch%args.adaptive_freq==0) & (epoch!=0):
            test_points, tMin_test, pMax_test = generate_training_points_7D(num_test_points)
            # test_points= test_points.detach().cpu()
            domain_points = adaptive_resample_points(
                model, test_points, 
                k=args.adaptive_k, c=args.adaptive_c, 
                num_points_to_add=len(domain_points),  # Use actual domain points length
                batch_size=args.batch_size
            )
            tMin_points = adaptive_resample_points(
                model, tMin_test, 
                k=args.adaptive_k, c=args.adaptive_c, 
                num_points_to_add=len(tMin_points),  # Use actual tMin points length
                batch_size=args.batch_size
            )
            pMax_points = adaptive_resample_points(
                model, pMax_test, 
                k=args.adaptive_k, c=args.adaptive_c, 
                num_points_to_add=len(pMax_points),  # Use actual pMax points length
                batch_size=args.batch_size
            )
            domain_points_total = domain_points

    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch 7D PDE Solver')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--model_path', type=str, default='pytorch_model_7D.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=1_000_000, help='Batch size for batched processing')
    parser.add_argument('--adaptive_sampling', action='store_true', help='Enable adaptive sampling (overrides --no_adaptive). Use: --adaptive_sampling --adaptive_freq 250')
    parser.add_argument('--adaptive_k', type=float, default=2.0, help='Power k for adaptive resampling')
    parser.add_argument('--adaptive_c', type=float, default=1.0, help='Constant c for adaptive resampling')
    parser.add_argument('--adaptive_freq', type=int, default=1000, help='Frequency of adaptive resampling (epochs)')
    # parser.add_argument('--adaptive_freq', type=int, default=2500, help='Frequency of adaptive resampling (epochs)')

    args = parser.parse_args()

    train_model_7D(args)
if __name__ == "__main__":
    main()