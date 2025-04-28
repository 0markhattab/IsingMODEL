# plots.py

import configparser
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.colors import ListedColormap

import ising  # ising.metropolis_sweep or wolff_cluster_sweep

def plot_energy_vs_temperature_separate(temperatures, energies, out_file):
    """
    Plots Energy vs. Temperature in a single separate figure.
    """
    plt.figure(figsize=(6,4))
    plt.plot(temperatures, energies, 'o', color='darkred')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Energy")
    plt.title("Energy vs Temperature (Wolff Algorithm)")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def plot_magnetization_vs_temperature_separate(temperatures, magnetizations, out_file):
    """
    Plots Magnetization vs. Temperature in a single separate figure.
    """
    plt.figure(figsize=(6,4))
    plt.plot(temperatures, magnetizations, 'o', color='blue')
    plt.xlabel("Temperature (T)")
    plt.ylabel("Magnetization")
    plt.title("Magnetization vs Temperature (Wolff Algorithm)")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def illustrate_evolution_magenta_yellow(lattice, sweep_function, beta, plot_times,
                                        out_file="images/snapshot_grid.png"):
    """
    Shows the lattice at selected timesteps: t = plot_times
    Using the specified sweep_function (metropolis_sweep or wolff_cluster_sweep)
    in a loop, then plotting the configuration in a grid.

    We create a custom colormap: -1 (spin down) => magenta, +1 (spin up) => yellow.
    """
    import math
    from matplotlib.colors import ListedColormap

    
    magenta_yellow = ListedColormap(["magenta", "yellow"])

    
    
    n_plots = len(plot_times)
    fig, axes = plt.subplots(1, n_plots, figsize=(3*n_plots, 3))

    
    if n_plots == 1:
        axes = [axes]

    
    current_time = 0
    for idx, tval in enumerate(plot_times):
        # do sweeps from 'current_time' up to 'tval'
        for step in range(current_time, tval):
            sweep_function(lattice, beta)
        current_time = tval

        
        ax = axes[idx]
        ax.imshow(lattice, vmin=-1, vmax=1, cmap=magenta_yellow)
        ax.set_title(f"t = {tval}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

def main(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

  
    path_time = config.get('paths','my_time')
    path_ene  = config.get('paths','my_ene')
    path_mag  = config.get('paths','my_mag')
    pic_time  = config.get('paths','time_pic')
    pic_enemag= config.get('paths','enemag_pic')


    energy_sep_pic = config.get('paths','energy_separate_pic')
    mag_sep_pic    = config.get('paths','mag_separate_pic')

    
    snapshot_met_pic   = config.get('paths','snapshots_met_pic')
    snapshot_wolff_pic = config.get('paths','snapshots_wolff_pic')

    N = config.getint('settings','N')
    M = config.getint('settings','M')
    numberTemp = config.getint('settings','numberTemp')
    startTemp  = config.getfloat('settings','startTemp')
    endTemp    = config.getfloat('settings','endTemp')


    if (not os.path.exists(path_ene)) or (not os.path.exists(path_mag)):
        print("Energy or magnetization data not found. Run simulation first!")
        sys.exit(1)

    energies = np.load(path_ene)       # shape = (numberTemp,)
    magnetizations = np.load(path_mag) # shape = (numberTemp,)
    temperatures = np.linspace(startTemp, endTemp, numberTemp)

  
    fig, ax1 = plt.subplots(figsize=(7,5))
    color1 = 'tab:red'
    ax1.set_xlabel('Temperature (T)')
    ax1.set_ylabel('Energy', color=color1)
    ax1.plot(temperatures, energies, 'o-', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Magnetization', color=color2)
    ax2.plot(temperatures, magnetizations, 's--', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    plt.title("Energy & Magnetization vs T (Wolff Algorithm)")
    plt.savefig(pic_enemag)
    plt.close()
    print(f"Saved {pic_enemag}")

    
    if os.path.exists(path_time):
        time_data = np.load(path_time)
        plt.figure(figsize=(7,5))
        plt.plot(time_data, marker='o', color='black')
        plt.xlabel("Step index")
        plt.ylabel("Magnetization per spin")
        plt.title("Time Evolution at Single Temperature")
        plt.savefig(pic_time)
        plt.close()
        print(f"Saved {pic_time}")
    else:
        print("No time-evolution data found. Skipping time evolution plot.")

   
    plot_energy_vs_temperature_separate(temperatures, energies, out_file=energy_sep_pic)
    plot_magnetization_vs_temperature_separate(temperatures, magnetizations, out_file=mag_sep_pic)


    T_demo = 2.5
    beta_demo = 1.0/T_demo
    plot_times = [0, 10, 100, 200, 500, 1000]

    # Metropolis snapshots
    lattice_met = ising.init_lattice(N, M, "random")
    illustrate_evolution_magenta_yellow(lattice_met,
                                        sweep_function=ising.metropolis_sweep,
                                        beta=beta_demo,
                                        plot_times=plot_times,
                                        out_file=snapshot_met_pic)

    # Wolff snapshots
    lattice_wolff = ising.init_lattice(N, M, "random")
    illustrate_evolution_magenta_yellow(lattice_wolff,
                                        sweep_function=ising.wolff_cluster_sweep,
                                        beta=beta_demo,
                                        plot_times=plot_times,
                                        out_file=snapshot_wolff_pic)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python plots.py configuration.txt")
        sys.exit(1)
    main(sys.argv[1])
