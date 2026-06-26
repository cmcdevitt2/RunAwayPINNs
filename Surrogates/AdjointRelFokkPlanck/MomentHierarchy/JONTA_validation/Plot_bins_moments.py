import numpy as np
import os
# from matplotlib.colors import LogNorm
import scipy
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import h5py
plt.rcParams.update({'font.size': 22})


def _fdist_txt_to_Z_energy_xi(data_arr, ng, nxi):
    """
    Z[iE, ixi] with shape (ng, nxi).

    BinParticles saves 3 columns (E, xi, f) with f raveled from meshgrid(E, xi), i.e.
    C-order (nxi, ng) — use f.reshape(nxi, ng).T to get (ng, nxi).

    Legacy: single column of length ng*nxi in (ng, nxi) C order (row = energy bin).
    """
    if data_arr.ndim == 2 and data_arr.shape[1] >= 3:
        f = np.ascontiguousarray(data_arr[:, 2]).ravel()
    else:
        f = np.ravel(data_arr)
    if f.size != ng * nxi:
        raise ValueError(
            f"fdist length {f.size} != ng*nxi ({ng}*{nxi}); got array shape {data_arr.shape}"
        )
    if data_arr.ndim == 2 and data_arr.shape[1] >= 3:
        return f.reshape(nxi, ng).T
    return f.reshape(ng, nxi)


def plot_distribution(file_path,t):
    """
    Loads particle distribution data from a text file and creates a plot.

    The text file is expected to have at least two columns of numerical data,
    separated by whitespace. The first column is treated as the x-axis
    (e.g., energy bin) and the second column as the y-axis (e.g., count or
    distribution value).

    Args:
        file_path (str): The full path to the data file.
    """
    try:
        # Load the data from the text file using numpy
        data = np.loadtxt(file_path)

        # Ensure data has at least two columns
        # if data.ndim < 2 or data.shape[1] < 2:
        #     print(f"Error: Data in {file_path} must have at least two columns.")
        #     return

        # Extract x and y data from the first two columns
        # x = data[:, 0]
        # y = data[:, 1]

        # Create a figure and an axes object
        fig, ax = plt.subplots(1,1)

        # Construct paths for the node files, assuming they are in the same directory
        base_dir = os.path.dirname(file_path)
        g_nodes_path = os.path.join(base_dir, 'g_nodes.txt')
        xi_nodes_path = os.path.join(base_dir, 'xi_nodes.txt')

        # Load the coordinate arrays
        g_nodes = np.loadtxt(g_nodes_path)
        energy_nodes= 0.511*(g_nodes-1)
        xi_nodes = np.loadtxt(xi_nodes_path)

        Z = _fdist_txt_to_Z_energy_xi(data, len(energy_nodes), len(xi_nodes)).T

        # Create a meshgrid for the contour plot axes
        X, Y = np.meshgrid(energy_nodes, xi_nodes)
        # print(X.shape)
        X_reshape=energy_nodes.reshape(1,len(energy_nodes))
        # print(Z.shape)
        # Create the filled contour plot
        # A logarithmic color scale is often useful for distributions
        contour = ax.contourf(X, Y,Z, levels=50, cmap='jet')

        # Add a color bar to the plot to show the scale of the distribution values
        fig.colorbar(contour, ax=ax,)

        # Set plot titles and labels for clarity
        ax.set_title(r'$t/\tau_c$='+ f"{float(t)/1000}")
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel(r'$\xi$')

        # Use a logarithmic scale for the y-axis, which is common for
        # distribution functions to show a wide dynamic range.
        # ax.set_yscale('log')

        # Add a grid for better readability
        # ax.grid(True, which="both", linestyle='--', linewidth=0.5)

        # Add a legend
        # ax.legend()

        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()

        # Save the plot to a file
        output_filename = f'plots/fDist_step_{t}.png'
        plt.savefig(output_filename)
        print(f"Plot saved as {output_filename}")

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_full_distribution(file_path, t,E0,xi0,Ebar,Zeff,alpha):
    """
    Save the full 2D distribution f(energy, xi) to a .txt file using the same
    load/reshape logic as plot_distribution. Output: data/full_distrib_{t}.txt
    with columns: energy [MeV], xi, f(energy, xi).
    """
    try:
        data = np.loadtxt(file_path)
        base_dir = os.path.dirname(file_path)
        g_nodes = np.loadtxt(os.path.join(base_dir, 'g_nodes.txt'))
        energy_nodes = 0.511 * (g_nodes - 1)
        xi_nodes = np.loadtxt(os.path.join(base_dir, 'xi_nodes.txt'))
        # n0 = get_init_density()
        Z = _fdist_txt_to_Z_energy_xi(data, len(energy_nodes), len(xi_nodes)).T  # (n_xi, n_energy)
        X, Y = np.meshgrid(energy_nodes, xi_nodes)
        os.makedirs('data', exist_ok=True)
        out_path = f'data/full_distrib_{t}_E0_{E0}_xi0_{xi0}_Ebar_{Ebar}_Zeff_{Zeff}_alpha_{alpha}.txt'
        np.savetxt(out_path, np.column_stack((X.ravel(), Y.ravel(), Z.ravel()/n0)),
                   header='energy_MeV xi f', comments='')
        print(f"Saved full distribution to {out_path}")
    except FileNotFoundError as e:
        print(f"Error: File not found — {e}")
    except Exception as e:
        print(f"Error saving full distribution: {e}")

def plot_energy_distribution(file_path,t):
    """
    Loads particle distribution data from a text file and creates a plot.

    The text file is expected to have at least two columns of numerical data,
    separated by whitespace. The first column is treated as the x-axis
    (e.g., energy bin) and the second column as the y-axis (e.g., count or
    distribution value).

    Args:
        file_path (str): The full path to the data file.
    """
    # try:
    # Load the data from the text file using numpy
    data = np.loadtxt(file_path)

    # Ensure data has at least two columns
    # if data.ndim < 2 or data.shape[1] < 2:
    #     print(f"Error: Data in {file_path} must have at least two columns.")
    #     return

    # Extract x and y data from the first two columns
    # x = data[:, 0]
    # y = data[:, 1]

    # Create a figure and an axes object
    fig, ax = plt.subplots()

    # Construct paths for the node files, assuming they are in the same directory
    base_dir = os.path.dirname(file_path)
    g_nodes_path = os.path.join(base_dir, 'g_nodes.txt')
    xi_nodes_path = os.path.join(base_dir, 'xi_nodes.txt')

    # Load the coordinate arrays
    g_nodes = np.loadtxt(g_nodes_path)
    p_nodes= np.sqrt(g_nodes**2-1)
    energy_nodes= 0.511*(g_nodes-1)
    xi_nodes = np.loadtxt(xi_nodes_path)

    Z = _fdist_txt_to_Z_energy_xi(data, len(energy_nodes), len(xi_nodes))
    
    n0 = get_init_density()
    Z_xi_integrate = scipy.integrate.simpson(Z,axis=1,x=xi_nodes)*p_nodes*g_nodes/n0
    ax.plot(energy_nodes, Z_xi_integrate)

    np.savetxt(f'data/energy_distrib_{t}.txt',(energy_nodes,Z_xi_integrate))
    # Set plot titles and labels for clarity
    ax.set_title('Particle Distribution')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel(r'$p^2 f_{RE}$')
    # ax.legend()
    plt.tight_layout()

    # Save the plot to a file
    output_filename = f'plots/energy_distrib_{t}.png'
    plt.savefig(output_filename)
    # print(f"Plot saved as {output_filename}")

    # Display the plot
    plt.show()

    return

def get_moments(file_path, t, gamma_threshold=-np.inf):
    # print(Z.shape)
    base_dir = os.path.dirname(file_path)
    g_nodes_path = os.path.join(base_dir, 'g_nodes.txt')
    xi_nodes_path = os.path.join(base_dir, 'xi_nodes.txt')

    # Load the coordinate arrays
    # g_nodes = np.loadtxt(g_nodes_path)
    # p_nodes = np.sqrt(g_nodes**2-1)
    # energy_nodes= 0.511*(g_nodes-1)
    # xi_nodes = np.loadtxt(xi_nodes_path)
    # g_nodes_reshape = g_nodes.reshape(len(energy_nodes),1)
    # p_nodes_reshape = p_nodes.reshape(len(energy_nodes),1)
    # v_nodes_reshape = p_nodes_reshape/g_nodes_reshape
    # # xi_nodes_reshape = xi_nodes.reshape(1,len(xi_nodes))
    # data = np.loadtxt(file_path)
    # Z = data.reshape(len(energy_nodes),len(xi_nodes))
    # n_p_integrate = scipy.integrate.simpson(Z*p_nodes_reshape*g_nodes_reshape,axis=0,x= g_nodes_reshape)
    # j_p_integrate = scipy.integrate.simpson(Z*p_nodes_reshape*p_nodes_reshape,axis=0,x= g_nodes_reshape)
    # E_p_integrate = scipy.integrate.simpson(Z*p_nodes_reshape*g_nodes_reshape*0.511*(g_nodes_reshape-1),axis=0,x= g_nodes_reshape)
    # pres_p_integrate = scipy.integrate.simpson(Z*v_nodes_reshape*p_nodes_reshape**2,axis=0,x= g_nodes_reshape)
    
    # n_total = 2*np.pi* scipy.integrate.simpson(n_p_integrate,x=xi_nodes)
    # j_total = 2*np.pi* scipy.integrate.simpson(j_p_integrate*xi_nodes,x=xi_nodes)
    # E_total = 2*np.pi* scipy.integrate.simpson(E_p_integrate,x=xi_nodes)
    # pressure_total = 2*np.pi* scipy.integrate.simpson(pres_p_integrate*0.5*(3*xi_nodes**2-1),x=xi_nodes)
    # gammas = []
    # xis    = []
    # for step in np.arange(args.start_time,args.end_time+args.dt,args.dt):
    # step = 500_000
    # for ddir in dev_dirs:
        # fpath = ddir / f"step_{step:06d}.h5"
    fpath = f"{file_path}/device_0/step_{t:06d}.h5"
    with h5py.File(fpath, "r") as f:
        gamma  = f["gamma"][:]
        xi  = f["xi"][:]
        # ids= f["ids"][:]
    # keep only alive particles
    # m_alive = (ids >= 0)
    # if m_alive.any():
        # gammas.append(g)
        # xis.append(x)

    ####Counting above pRE 
    gamma = np.asarray(gamma)
    p = np.sqrt(gamma**2-1)
    v = np.asarray(p/gamma)
    xi = np.asarray(xi)
    gamma_ok = np.where(
        np.isfinite(gamma) & (gamma > gamma_threshold),
        True,
        False,
    )
    n_total = int(np.sum(gamma_ok))
    v_ok = np.where(gamma_ok, v,np.nan)
    xi_j = np.where(
        gamma_ok & np.isfinite(xi) & (xi >= -1.0) & (xi <= 1.0),
        xi,
        np.nan,
    )
    j_total = float(np.nansum(xi_j*v_ok))
    E_total = float(
        0.511 * np.nansum(np.where(gamma_ok, gamma - 1.0, np.nan))
    )
    pressure_total = float(
        np.nansum(np.where(gamma_ok, v_ok**2 * 0.5 * (3.0 * xi**2 - 1.0), np.nan))
    )



    ####Terminal condition method##
    # pRE = np.sqrt((1.0/0.511+1)**2-1)
    # pMax = np.sqrt((16/0.511+1)**2-1)
    # pMin = np.sqrt((0.01/0.511+1)**2-1)
    # p = np.sqrt(gamma**2-1)

    # N_current,N_pressure,N_energy = 80,80,32
    # Prob_current = (pRE - p) / (pMax - pMin) * N_current
    # Prob_pressure = (pRE - p) / (pMax - pMin) * N_pressure
    # Prob_energy = (pRE - p) / (pMax - pMin) * N_energy
    
    # P_terminal_current = 0.5 * (1.0 - torch.tanh(Prob_current))
    # P_terminal_pressure = 0.5 * (1.0 - torch.tanh(Prob_pressure))
    # P_terminal_energy = 0.5 * (1.0 - torch.tanh(Prob_energy))

    # j_total = np.nansum()

    return n_total,j_total,E_total,pressure_total

def get_an_energy_distrib(file_path,n):
    base_dir = os.path.dirname(file_path)
    g_nodes_path = os.path.join(base_dir, 'g_nodes.txt')
    xi_nodes_path = os.path.join(base_dir, 'xi_nodes.txt')

    # Load the coordinate arrays
    g_nodes = np.loadtxt(g_nodes_path)
    p_nodes = np.sqrt(g_nodes**2-1)
    energy_nodes= 0.511*(g_nodes-1)
    xi_nodes = np.loadtxt(xi_nodes_path)
    g_nodes_reshape = g_nodes.reshape(len(energy_nodes),1)
    p_nodes_reshape = p_nodes.reshape(len(energy_nodes),1)
    v_nodes_reshape = p_nodes_reshape/g_nodes_reshape
    # xi_nodes_reshape = xi_nodes.reshape(1,len(xi_nodes))
    data = np.loadtxt(file_path)
    Z = _fdist_txt_to_Z_energy_xi(data, len(energy_nodes), len(xi_nodes))
    # print(Z)
    # exit()
    # print(np.max(Z[0,:]),np.min(Z[0,:]))
    legendre_values = scipy.special.eval_legendre(n, xi_nodes)
    leg_values_reshape = legendre_values.reshape(1,len(xi_nodes))

    # an_p_integrate = scipy.integrate.simpson(Z*p_nodes_reshape*g_nodes_reshape*,axis=0,x= g_nodes_reshape)
    an_energy_distrib = scipy.integrate.simpson(Z*leg_values_reshape,x=xi_nodes,axis=1)
    n0 = get_init_density()    
    return energy_nodes, an_energy_distrib/n0

def save_an_energy_distribs(file_path, t, max_n=10):
    """
    Save energy distributions for each Legendre order n up to max_n at a given time.
    
    Args:
        file_path (str): Path to the distribution data file
        t (str or int): Time step identifier (e.g., '001000' or 1000)
        max_n (int): Maximum Legendre order (default: 10)
    """
    os.makedirs('data', exist_ok=True)
    for n in range(max_n + 1):
        energy_nodes, an_distrib = get_an_energy_distrib(file_path, n)
        filename = f'data/f_{n}_energy_{t}.txt'
        np.savetxt(filename, (energy_nodes, an_distrib))
        print(f"Saved {filename}")
def get_init_density():
    file_path = 'particles/fDist_step_000000.txt'
    base_dir = os.path.dirname(file_path)

    xi_nodes_path = os.path.join(base_dir, 'xi_nodes.txt')
    xi_nodes = np.loadtxt(xi_nodes_path)

    g_nodes_path = os.path.join(base_dir, 'g_nodes.txt')
    g_nodes = np.loadtxt(g_nodes_path)
    p_nodes = np.sqrt(g_nodes**2-1)
    energy_nodes= 0.511*(g_nodes-1)
    data = np.loadtxt(file_path)
    p_nodes_reshape = p_nodes.reshape(len(energy_nodes),1)

    g_nodes_reshape = g_nodes.reshape(len(energy_nodes),1)
    Z = _fdist_txt_to_Z_energy_xi(data, len(energy_nodes), len(xi_nodes))

    Z_p_integrate = 2*np.pi*scipy.integrate.simpson(Z*p_nodes_reshape*g_nodes_reshape,axis=0,x= g_nodes_reshape)
    # print(Z_p_integrate.shape)
    # Count nonzero values
    # nonzero_count = np.count_nonzero(Z_p_integrate)
    
    # xi_init=len(Z_p_integrate)/nonzero_count
    # Z_xi_integrate = 
    Z_xi_integrate = scipy.integrate.simpson(Z_p_integrate,axis=0,x=xi_nodes)
    
    # Z_xi_integrate1 = scipy.integrate.simpson(Z_p_integrate,axis=0,x=xi_nodes)
    # Z_xi_integrate1 =0
    # Z_xi_integrate2 = scipy.integrate.simpson(Z_p_integrate,axis=0,x=xi_nodes)
    # xi_init = 1/((Z_xi_integrate1/Z_xi_integrate)+1)
    # Z_ones=np.sum(np.ones_like(Z_p_integrate))/len(xi_nodes)
    # print(Z_ones)
    # print(Z_p_integrate)
    return Z_xi_integrate

def parse_args():
    parser = argparse.ArgumentParser(description='Plot particle distribution data from JAX simulation output')
    parser.add_argument('--run_id', type=str, default='.', 
                       help='Run identifier for input directory')
    parser.add_argument('--base_dir', type=str, default='particles', 
                       help='Base directory containing run data (default: ./particles)')
    parser.add_argument('--start_time', type=int, default=0, 
                       help='Start time for processing (default: 0.0)')
    parser.add_argument('--end_time', type=int, default=20_000, 
                       help='End time for processing (default: 20.0)')
    parser.add_argument('--dt', type=int, default=1_000, 
                       help='Time step size used in simulation (default: 1e-3)')
    parser.add_argument('--E0', type=float, default=5.0)
    parser.add_argument('--xi0', type=float, default=0.0)
    parser.add_argument('--Ebar', type=float, default=0.0)
    parser.add_argument('--Z_eff', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--N', type=int, default=1_000_000, 
                       help='Time step size used in simulation (default: 1e-3)')
    parser.add_argument('--Energy_threshold', type=float, default=float('-inf'),
                       help='Only count particles with energy above this threshold')
                       
    parser.add_argument('--plot_pinned', action='store_true',
                       help='Plot xi distribution of pinned particles using h5 files')
    parser.add_argument('--h5_only', action='store_true',
                       help='Only plot pinned particles, skip other plots')
    return parser.parse_args()
if __name__ == '__main__':
    # Define the path to the data file
    args = parse_args()

    run_dir = Path(args.base_dir) / args.run_id
    # out_dir = args.base_dir + "/" + args.run_id

    fig, ax = plt.subplots()
    x_arr= np.array([])
    n_arr = np.array([])
    j_arr = np.array([])
    E_arr = np.array([])
    press_arr = np.array([])
    pinned_stats = []
    gamma_threshold = args.Energy_threshold/0.511+1
    for t in np.arange(args.start_time,args.end_time+args.dt,args.dt):
        data_file_path = f'particles/fDist_step_{t:06d}.txt'
        plot_energy_distribution(data_file_path,f'{t:06d}')
        plot_distribution(data_file_path,f'{t:06d}')
        n_t, j_t, E_t, pres_t = get_moments(
            run_dir, t, gamma_threshold=gamma_threshold
        )
 
        n_arr = np.append(n_arr,n_t)
        j_arr = np.append(j_arr,j_t)
        E_arr = np.append(E_arr,E_t)
        press_arr = np.append(press_arr,pres_t)
    # Save n_arr and j_arr to text files (only if not h5_only mode)
    if not args.h5_only:
        time_arr = np.arange(args.start_time, args.end_time+args.dt, args.dt)
        np.savetxt(f'data/n_vs_time.txt', np.column_stack((time_arr, n_arr/args.N)), header='time n')
        np.savetxt(f'data/j_vs_time.txt', np.column_stack((time_arr, -j_arr/args.N)), header='time j')
        np.savetxt(f'data/E_vs_time.txt', np.column_stack((time_arr, E_arr/args.N)), header='time j')
        np.savetxt(f'data/pressure_vs_time.txt', np.column_stack((time_arr, press_arr/args.N)), header='time j')
