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
import os
import skopt
import argparse
import gc
import scipy
import tqdm

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
# Alpha range (NEW!)
alphaMin, alphaMax = 0.05, 0.2  # Variable radiation parameter range
# pRE is now a constant
gRE = 1+1e6/mecSQ
pRE_constant = np.sqrt(gRE**2-1)  # Fixed pRE value

print(f"pMax: {pMax}, pMin: {pMin}")
print(f"EF range: {EFMin} to {EFMax}")
print(f"Zeff range: {ZeffMin} to {ZeffMax}")
print(f"Alpha range: {alphaMin} to {alphaMax}")
print(f"pRE constant: {pRE_constant}")

# Training parameters
lr = 5.e-4
epochs = 500_000
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

# Hammersly sampling
def hammersley_sequence(n_samples, dim):
    skip = 0
    if dim == 1:
        sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
    else:
        sampler = skopt.sampler.Hammersly()
        skip = 1
    space = [(0.0, 1.0)] * dim
    points = np.asarray(sampler.generate(space, n_samples + skip)[skip:])
    return points

def output_transform_6D(inputs, outputs):
    """Output transform function for 6D problem"""
    pNorm, xiNorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6]
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    EF = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin

    # pRE is now constant
    pRE = pRE_constant
    Prob = torch.tanh(pNorm*outputs[:,0:1])
    return Prob

def pde_residual_6D(inputs, outputs):
    """PDE residual function for 6D problem"""
    pNorm, xiNorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6]
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    # Use variable EF, Zeff, and alpha values
    EFval = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin
    
    ones = torch.ones_like(outputs)
    # First derivatives
    dy_P = grad(outputs, inputs, grad_outputs=ones,create_graph=True)[0]
    dy_p = dy_P[:, 0:1]/(pMax-pMin)
    dy_xi = dy_P[:,1:2]/(xiMax-xiMin)
    dy_t = dy_P[:,2:3]/(tMax-tMin)
    # Second derivatives
    dy_xixi = grad(dy_xi, inputs,grad_outputs=ones, create_graph=True)[0][:, 1:2]/(xiMax-xiMin)
    gamma = torch.sqrt(1 + p * p)
    
    Ephi = -EFval
    
    ElectricFieldTerms = -Ephi * (xi * dy_p + ((1 - xi**2) / p) * dy_xi)
    CollisionalTerms = (gamma * gamma / p**2) * dy_p - ((Zeff + 1) / 2) * (gamma / p**3) * ((1 - xi**2) * dy_xixi - 2 * xi * dy_xi)
    RadiationTerms = alpha * (gamma * p * (1 - xi**2) * dy_p - xi * (1 - xi**2) / gamma * dy_xi)


    loss =(p**2/(1+p**2))*(dy_t + ElectricFieldTerms + CollisionalTerms + RadiationTerms)
    return loss

def tMin_func_6D(X):
    """Initial condition function for 6D"""
    pNorm, xiNorm, tNorm, EFNorm, ZeffNorm, alphaNorm = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4], X[:, 4:5], X[:, 5:6]
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    EF = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin

    # pRE is now constant
    pRE = pRE_constant
    N = 80
    Prob = (pRE - p) / (pMax - pMin) * N
    v = p/torch.sqrt(1+p**2)
    Pinit =-v*xi* 0.5 * (1.0 - torch.tanh(Prob))
    return Pinit

def pMax_func_6D(inputs):
    """Function that defines what the solution should be at pMax boundary - 6D version
    Boundary condition: solution = 1 when p = pMax and xi = -1"""
    pNorm, xiNorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6]
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    # Use variable EF, Zeff, and alpha values
    EFval = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin

    # Check if points are at pMax boundary
    at_pMax = torch.isclose(p, torch.tensor(pMax, dtype=dtype, device=device), atol=1e-6)
    
    Up = -EFval * xi - (1 + p**2) / (p**2) - alpha * p * torch.sqrt(1 + p**2) * (1 - xi**2)
    runaway_mask = Up > 0
    
    # Combine both conditions: must be at pMax AND in runaway region
    combined_mask = at_pMax & runaway_mask
    
    # Solution value is 1 at this boundary
    # solution_value = torch.ones_like(combined_mask, dtype=dtype, device=device)
    gamma = np.sqrt(pMax**2 + 1)
    v = pMax/gamma
    solution_value = -v*xi
    # Return the solution value only where the boundary condition applies
    return torch.where(combined_mask, solution_value, torch.zeros_like(combined_mask, dtype=dtype, device=device))

def pMax_bound_6D(inputs):
    """Boundary condition for pMax - 6D version
    Returns mask for points where p = pMax and xi = -1"""
    pNorm, xiNorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6]
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    # Use variable EF, Zeff, and alpha values
    EFval = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin

    # Check if points are at pMax boundary
    at_pMax = torch.isclose(p, torch.tensor(pMax, dtype=dtype, device=device), atol=1e-6)
    
    # Check runaway condition: Up > 0
    Up = -EFval * xi - (1 + p**2) / (p**2) - alpha * p * torch.sqrt(1 + p**2) * (1 - xi**2)
    runaway_mask = Up > 0
    # Combine both conditions: must be at pMax AND xi = -1
    combined_mask = at_pMax & runaway_mask
    return combined_mask.squeeze()

def tMin_bound_6D(inputs):
    """Boundary condition for tMin - 6D version"""
    pNorm, xiNorm, tNorm, EFNorm, ZeffNorm, alphaNorm = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5], inputs[:, 5:6]
    p = (pMax-pMin)*pNorm+pMin
    xi = (xiMax-xiMin)*xiNorm+xiMin
    t = (tMax-tMin)*tNorm+tMin
    EF = (EFMax-EFMin)*EFNorm+EFMin
    Zeff = (ZeffMax-ZeffMin)*ZeffNorm+ZeffMin
    alpha = (alphaMax-alphaMin)*alphaNorm+alphaMin
    
    # Check if points are at tMin (t=0)
    mask = torch.isclose(t, torch.tensor(tMin, dtype=dtype, device=device), atol=1e-6)
    return mask.squeeze()

def generate_training_points_6D(num_domain_points):
    """Generate training points for 6D domain and boundary - now with (p, xi, t, EF, Zeff, alpha)"""
    # Domain points - 6D
    engine = SobolEngine(dimension=6, scramble=True, seed=seed)
    domain_points = engine.draw(num_domain_points,dtype=dtype)
    domain_points=domain_points.to(device).requires_grad_(True)
    # Generate boundary points and add to domain points
    num_boundary_per_type = num_domain_points // 20  # 5% of domain points per boundary type
    
    # xi=-1 boundary points
    xi_minus1_points = engine.draw(num_boundary_per_type, dtype=dtype).to(device)
    xi_minus1_points[:, 1] = 0  # Set xi to xiMin (-1) in normalized coordinates
    xi_minus1_points = xi_minus1_points.requires_grad_(True)
    # # xi=1 boundary points
    xi_plus1_points = engine.draw(num_boundary_per_type, dtype=dtype).to(device)
    xi_plus1_points[:, 1] = 1  # Set xi to xiMax in normalized coordinates
    xi_plus1_points = xi_plus1_points.requires_grad_(True)
    domain_points = torch.cat([domain_points, xi_minus1_points, xi_plus1_points], dim=0).requires_grad_(True)
    total_boundary_added = 2*num_boundary_per_type
    print(f"Added {total_boundary_added} boundary points to domain:")
    print(f"  - {num_boundary_per_type} xi=-1 points")
    print(f"  - {num_boundary_per_type} pMin points") 
    print(f"Total domain points: {len(domain_points)}")

    # Boundary points for pMax - 6D
    # These points must be at p = pMax AND Up>0
    pMax_points = engine.draw(num_boundary_points // 2,dtype=dtype)
    pMax_points[:, 0] = 1  # Set p to pMax (normalized)
    
    # Convert to physical coordinates
    alpha_norm_pMax = pMax_points[:, 5]
    alpha_pMax = alpha_norm_pMax * (alphaMax - alphaMin) + alphaMin
    E_norm_pMax = pMax_points[:, 3]
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
    
    # print(f"Xi root statistics: min={np.min(xiMax_pMax):.3f}, max={np.max(xiMax_pMax):.3f}, mean={np.mean(xiMax_pMax):.3f}")
    # # print(f"Valid solutions: {np.sum(valid1 | valid2)}/   {len(xiMax_pMax)}")
    
    # Sample xi in the range [-1, xiMax_pMax] where Up > 0
    xiMax_norm_pMax = (xiMax_pMax - xiMin) / (xiMax - xiMin)
    
    # Set the new xi values
    pMax_points[:, 1] = pMax_points[:, 1]  * xiMax_norm_pMax
    pMax_points = pMax_points.to(device).requires_grad_(True)
    
    # Boundary points for tMin - 6D (dimension index 2)
    tMin_points= engine.draw(num_boundary_points//2,dtype=dtype)
    tMin_points[:, 2] = 0  # Set t to tMin (dimension index 2: p, xi, t, EF, Zeff, alpha)
    tMin_points=tMin_points.to(device).requires_grad_(True)
    
    # Combine both boundary point types
    boundary_points = torch.cat([pMax_points, tMin_points], dim=0).requires_grad_(True)
    
    print(f"Boundary conditions:")
    print(f"  - pMax boundary: {len(pMax_points)} points at p=pMax where Up>0 (runaway region)")
    print(f"  - tMin boundary: {num_boundary_points // 2} points at t=0 (initial condition)")
    
    return domain_points, boundary_points

def compute_losses_6D(model, domain_points, boundary_points, weight_network=None):
    """Compute all loss components for 6D problem with network-based adaptive weighting"""
    # PDE loss
    domain_points = domain_points.requires_grad_(True)
    domain_outputs = model(domain_points)
    domain_outputs_transformed = output_transform_6D(domain_points, domain_outputs)
    resid = pde_residual_6D(domain_points, domain_outputs_transformed)
    
    # Boundary condition losses
    boundary_points = boundary_points.requires_grad_(True)
    boundary_outputs = model(boundary_points)
    boundary_outputs_transformed = output_transform_6D(boundary_points, boundary_outputs)
    
    # pMax boundary condition
    pMax_mask = pMax_bound_6D(boundary_points)
    if pMax_mask.any():
        pMax_target = pMax_func_6D(boundary_points[pMax_mask])
        pMax_loss = boundary_outputs_transformed[pMax_mask] - pMax_target
    else:
        pMax_loss = torch.tensor(0.0, dtype=dtype, device=device)
    
    # tMin boundary condition
    tMin_mask = tMin_bound_6D(boundary_points)
    if tMin_mask.any():
        tMin_target = tMin_func_6D(boundary_points[tMin_mask])
        tMin_loss = boundary_outputs_transformed[tMin_mask] - tMin_target
    else:
        tMin_loss = torch.tensor(0.0, dtype=dtype, device=device)
    
    return resid, pMax_loss, tMin_loss

def compute_losses_batched_6D(model, domain_points, boundary_points, batch_size=10000, weight_network=None, use_weights=True):
    """Compute all loss components with batched processing for 6D"""
    # PDE loss - process in batches
    total_pde_loss = 0.0
    num_batches = (len(domain_points) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(domain_points))
        batch_points = domain_points[start_idx:end_idx].detach().requires_grad_(True)
        
        batch_outputs = model(batch_points)
        batch_outputs_transformed = output_transform_6D(batch_points, batch_outputs)
        batch_resid = pde_residual_6D(batch_points, batch_outputs_transformed)
        
        batch_pde_loss = torch.mean(batch_resid**2)
        
        total_pde_loss += batch_pde_loss.item() * (end_idx - start_idx)
    
    pde_loss = torch.tensor(total_pde_loss / len(domain_points), dtype=dtype, device=device)
    
    # Boundary condition losses
    boundary_points = boundary_points.requires_grad_(True)
    pMax_mask = pMax_bound_6D(boundary_points)
    boundary_outputs = model(boundary_points)
    boundary_outputs_transformed = output_transform_6D(boundary_points, boundary_outputs)
    
    # pMax boundary condition
    if pMax_mask.any():
        pMax_target = pMax_func_6D(boundary_points[pMax_mask])
        pMax_loss = torch.mean((boundary_outputs_transformed[pMax_mask] - pMax_target)**2)
    else:
        pMax_loss = torch.tensor(0.0, dtype=dtype, device=device)
    
    # tMin boundary condition
    tMin_mask = tMin_bound_6D(boundary_points)
    if tMin_mask.any():
        tMin_target = tMin_func_6D(boundary_points[tMin_mask])
        tMin_loss = torch.mean((boundary_outputs_transformed[tMin_mask] - tMin_target)**2)
    else:
        tMin_loss = torch.tensor(0.0, dtype=dtype, device=device)
    
    total_loss = pde_loss + pMax_loss + tMin_loss
    
    return total_loss, pde_loss, pMax_loss, tMin_loss


def plot_CPF_6D(model, Zeffgrid, alphagrid, tgrid, EFgrid, losses, domain_points):
    """Plot CPF for 6D problem - now with Zeff and alpha"""
    nump, numxi, numZeff, numalpha, numt, numEF = 100, 101, 1, 1, 1, 1
    
    pgrid = np.linspace(0, 1, nump)
    pgridreal = np.linspace(pMin, pMax, nump)
    p_real_to_eV = np.array([(np.sqrt(p**2 + 1) - 1) * 511e3 for p in pgridreal])
    
    xigrid = np.linspace(0, 1, numxi)
    xigrid_real=np.linspace(xiMin,xiMax,numxi)
    
    # Create meshgrid for 6D - now (p, xi, t, EF, Zeff, alpha)
    pnew, xinew, tnew, EFnew, Zeffnew, alphanew = np.meshgrid(pgrid, xigrid, tgrid, EFgrid, Zeffgrid, alphagrid)
    X = np.vstack((np.ravel(pnew), np.ravel(xinew), 
                   np.ravel(tnew), np.ravel(EFnew), np.ravel(Zeffnew), np.ravel(alphanew))).T
    
    # Convert to torch tensor
    X_torch = torch.tensor(X, dtype=dtype, device=device)
    
    # Get predictions
    with torch.no_grad():
        y_pred = model(X_torch)
        y_pred_transformed = output_transform_6D(X_torch, y_pred)
    
    # Compute PDE residual
    X_torch_grad = X_torch.detach().requires_grad_(True)
    y_pred_grad = model(X_torch_grad)
    y_pred_transformed_grad = output_transform_6D(X_torch_grad, y_pred_grad)
    pde_res = pde_residual_6D(X_torch_grad, y_pred_transformed_grad)
    
    # Reshape for plotting - now (numxi, nump, numt, numEF, numZeff, numalpha)
    unew = y_pred_transformed.detach().cpu().numpy().reshape(numxi, nump, numt, numEF, numZeff, numalpha)
    unew_res = pde_res.detach().cpu().numpy().reshape(numxi, nump, numt, numEF, numZeff, numalpha)

    print('Test MSE=',f"{np.mean(unew_res**2)}")
    
    # Create plots
    fig0, (ax0) = plt.subplots(nrows=1, ncols=1)
    fig1, (ax1) = plt.subplots(nrows=1, ncols=1)
    fig0.set_tight_layout(True)
    fig1.set_tight_layout(True)
    # Plot CPF
    cs0 = ax0.contourf(p_real_to_eV / 1E6, xigrid_real, unew[:, :, 0, 0, 0, 0], 500, cmap='jet')
    # ax0.contour(p_real_to_eV / 1E6, xigrid_real, unew[:, :, 0, 0, 0, 0], levels=[0.5], colors='black')
    fig0.colorbar(cs0, ax=ax0)
    
    # Plot residual
    cs1 = ax1.contourf(p_real_to_eV / 1E6, xigrid_real, unew_res[:, :, 0, 0, 0, 0],500, cmap='jet')
    fig1.colorbar(cs1, ax=ax1)
    
    # Labels and titles
    ax0.set_xlabel("Energy [MeV]")
    ax1.set_xlabel("Energy [MeV]")
    ax0.set_ylabel("$\\xi$")
    ax1.set_ylabel("$\\xi$")
    EF_val = (EFMax-EFMin)*EFgrid+EFMin
    Zeff_val = (ZeffMax-ZeffMin)*Zeffgrid+ZeffMin
    alpha_val = (alphaMax-alphaMin)*alphagrid+alphaMin
    treal_str=tgrid*(tMax-tMin)+tMin
    # Ensure plots directory exists
    import os
    plots_dir = f'{dataout}/plots'
    os.makedirs(plots_dir, exist_ok=True)
    fig0.set_tight_layout(True)
    fig1.set_tight_layout(True)
    
    # Save plots
    EF_str = f'{EF_val:.2f}'.replace('.', 'o')
    Zeff_str = f'{Zeff_val:.2f}'.replace('.', 'o')
    alpha_str = f'{alpha_val:.4f}'.replace('.', 'o')
    tgrid_str = f'{tgrid}'.replace('.', 'o')
    fig0.savefig(f'{plots_dir}/CPF_6D_tgrid={tgrid_str}_EF={EF_str}_Zeff={Zeff_str}_alpha={alpha_str}_pytorch.png', dpi=300, bbox_inches='tight')
    fig1.savefig(f'{plots_dir}/CPF_6D_tgrid={tgrid_str}_EF={EF_str}_Zeff={Zeff_str}_alpha={alpha_str}_pytorch_res.png', dpi=300, bbox_inches='tight')
    plt.close(fig0)
    
    print(f"Plots saved: CPF_6D_tgrid={tgrid_str}_EF={EF_str}_Zeff={Zeff_str}_alpha={alpha_str}.png")
    return
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
        y_pred_transformed = output_transform_6D(batch_points, y_pred)
        pde_residuals = pde_residual_6D(batch_points, y_pred_transformed)
        batch_residuals = torch.abs(pde_residuals).detach().cpu().numpy()
        all_residuals.append(batch_residuals)
    
    residuals_abs = np.concatenate(all_residuals)
    
    # Calculate probability distribution
    resid_mean=np.abs(residuals_abs)/np.abs(residuals_abs).mean()
    err_eq_domain = np.power(resid_mean, k) + c
    
    # Flatten and normalize to get probability distribution
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
    
    return new_domain_points

def load_model(model_path):
    """Load a trained model from file"""
    model = FCNN(6,architecture).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def train_model_6D(args):
    """Main training function for 6D problem"""
    model = FCNN(6,architecture).to(device)
    # Print adaptive sampling status
    use_adaptive_sampling = args.adaptive_sampling
    if use_adaptive_sampling:
        print(f"Adaptive sampling ENABLED with frequency={args.adaptive_freq} epochs")
    else:
        print("Adaptive sampling DISABLED")
    optimizer = Soap(model.parameters(), lr=lr,betas=(0.997,0.997))
    print(f'model parameters= {sum(p.numel() for p in model.parameters())}')

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999995)
        
    # Generate training points
    domain_points, boundary_points = generate_training_points_6D(num_domain_points)
    test_points, boundary_points_test = generate_training_points_6D(num_test_points)
    test_points.detach().cpu()
    
    # Ensure data directory exists
    import os
    data_dir = f'{dataout}/data'
    os.makedirs(data_dir, exist_ok=True)
    
    pbar = tqdm.trange(epochs)

    total_test_loss_str ="--------"
    pMax_test_loss_str = "--------"
    tMin_test_loss_str = "--------"

    # Initialize loss history tracking for SOAP optimizer
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
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Compute losses - ensure fresh tensors to avoid graph reuse issues
        domain_points_fresh = domain_points.detach().requires_grad_(True)
        boundary_points_fresh = boundary_points.detach().requires_grad_(True)
        pde_resid, pMax_resid, tMin_resid = compute_losses_6D(model, domain_points_fresh, boundary_points_fresh)
        
        pde_loss = torch.mean(pde_resid**2)
        pMax_loss = torch.mean(pMax_resid**2)
        tMin_loss = torch.mean(tMin_resid**2)
            
        total_loss = pde_loss + pMax_loss + tMin_loss
        
        # Update main model
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record training losses for SOAP optimizer
        soap_train_loss_history.append(total_loss.item())
        soap_train_pde_loss_history.append(pde_loss.item())
        soap_train_bc_loss_history.append(pMax_loss.item())
        soap_train_ic_loss_history.append(tMin_loss.item())
        soap_epoch_history.append(epoch)
        soap_test_loss_history.append(0) # Placeholder
        soap_test_pde_loss_history.append(0) # Placeholder
        soap_test_bc_loss_history.append(0) # Placeholder
        soap_test_ic_loss_history.append(0) # Placeholder
        
        pbar.set_postfix({
            "Train PDE": f"{pde_loss.item():.2e}",
            "Train IC": f"{tMin_loss.item():.2e}",
            "Train BC": f"{pMax_loss.item():.2e}",
            "Test PDE": f"{total_test_loss_str}",
            "Test IC": f"{tMin_test_loss_str}",
            "Test BC": f"{pMax_test_loss_str}",
        })
        
        if epoch % args.adaptive_freq == 0:
            domain_points.detach().cpu()
            test_points.to(device)
            total_test_loss, pde_test_loss, pMax_test_loss, tMin_test_loss = compute_losses_batched_6D(model, test_points, boundary_points)
            test_points.detach().cpu()
            domain_points.to(device)
            total_test_loss_str = f"{pde_test_loss.item():.2e}"
            pMax_test_loss_str = f"{pMax_test_loss.item():.2e}" 
            tMin_test_loss_str = f"{tMin_test_loss.item():.2e}" 

            # Update test loss history for SOAP optimizer
            soap_test_loss_history[-1] = total_test_loss.item()
            soap_test_pde_loss_history[-1] = pde_test_loss.item()
            soap_test_bc_loss_history[-1] = pMax_test_loss.item()
            soap_test_ic_loss_history[-1] = tMin_test_loss.item()

            # Save SOAP optimizer loss history
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
            np.savetxt(f'{data_dir}/soap_loss_history_6d.csv', soap_loss_data, delimiter=',', header=soap_header, comments='')
            print(f"SOAP optimizer loss history saved to {data_dir}/soap_loss_history_6d.csv")

            # Plot for different EF, Zeff, and alpha values
            for EF_val in [2.0]:
                EF_norm = (EF_val - EFMin) / (EFMax - EFMin)
                for Zeff_val in [2.0]:
                    Zeff_norm = (Zeff_val - ZeffMin) / (ZeffMax - ZeffMin)
                    for alpha_val in [0.0, 0.1]:
                        alpha_norm = (alpha_val - alphaMin) / (alphaMax - alphaMin)
                        plot_CPF_6D(model, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, tgrid=tMin, EFgrid=EF_norm, losses=None, domain_points=boundary_points)
                        plot_CPF_6D(model, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, tgrid=0.5, EFgrid=EF_norm, losses=None, domain_points=domain_points)
                
            torch.save(model.state_dict(), f'{dataout}/pytorch_model_6D.pth')
            
        # Adaptive resampling
        use_adaptive_sampling = args.adaptive_sampling
        if use_adaptive_sampling and epoch > 0 and epoch % args.adaptive_freq == 0:
            print(f"\nPerforming adaptive resampling at epoch {epoch}...")
            test_points, boundary_points_test = generate_training_points_6D(num_test_points)
            domain_points = adaptive_resample_points(
                model, test_points, 
                k=args.adaptive_k, c=args.adaptive_c, 
                num_points_to_add=len(domain_points),
                batch_size=args.batch_size
            )
            boundary_points = adaptive_resample_points(
                model, boundary_points_test, 
                k=args.adaptive_k, c=args.adaptive_c, 
                num_points_to_add=len(boundary_points),  # Use actual boundary points length
                batch_size=args.batch_size
            )
    
    # Final plots
    for EF_val in [2.0, 2.5]:
        EF_norm = (EF_val - EFMin) / (EFMax - EFMin)
        for Zeff_val in [2.0]:
            Zeff_norm = (Zeff_val - ZeffMin) / (ZeffMax - ZeffMin)
            for alpha_val in [0.0, 0.1, 0.2]:
                alpha_norm = (alpha_val - alphaMin) / (alphaMax - alphaMin)
                plot_CPF_6D(model, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, tgrid=tMin, EFgrid=EF_norm, losses=None, domain_points=boundary_points)
                plot_CPF_6D(model, Zeffgrid=Zeff_norm, alphagrid=alpha_norm, tgrid=0.5, EFgrid=EF_norm, losses=None, domain_points=domain_points)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch 6D PDE Solver')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--plot', action='store_true', help='Generate plots from trained model')
    parser.add_argument('--model_path', type=str, default='pytorch_model_6D.pth', help='Path to trained model')
    parser.add_argument('--adaptive_sampling', action='store_true', help='Enable adaptive sampling')
    parser.add_argument('--adaptive_k', type=float, default=2.0, help='Power k for adaptive resampling')
    parser.add_argument('--adaptive_c', type=float, default=1.0, help='Constant c for adaptive resampling')
    parser.add_argument('--adaptive_freq', type=int, default=1000, help='Frequency of adaptive resampling (epochs)')
    parser.add_argument('--batch_size', type=int, default=1_000_000, help='Batch size for batched processing')
        
    args = parser.parse_args()
    
    if args.train:
        model = train_model_6D(args)        
    elif args.plot:
        model = FCNN(6,architecture).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"6D Model loaded from {args.model_path}")
        for tval in [0.0,0.5,1.0]:
            plot_CPF_6D(model, Zeffgrid = (3-ZeffMin)/(ZeffMax-ZeffMin), alphagrid= (0.1-alphaMin)/(alphaMax-alphaMin),tgrid=tval,EFgrid = (2.5-EFMin)/(EFMax-EFMin),losses=None,domain_points=None)
        
    else:
        print("Use --train flag to train the model, --plot flag to generate plots")

if __name__ == "__main__":
    main()

