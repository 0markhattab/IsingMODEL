# simulation.py

import configparser
import numpy as np
import sys
import os
import ising

def main(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    N = config.getint('settings','N')
    M = config.getint('settings','M')
    numberTemp = config.getint('settings','numberTemp')
    startTemp  = config.getfloat('settings','startTemp')
    endTemp    = config.getfloat('settings','endTemp')
    eqSteps    = config.getint('settings','eqSteps')
    measureSteps = config.getint('settings','measureSteps')
    timeEvolutionTemp = config.getfloat('settings','timeEvolutionTemp')
    algorithm = config.get('settings','algorithm').strip().lower()

    path_time = config.get('paths','my_time')
    path_ene  = config.get('paths','my_ene')
    path_mag  = config.get('paths','my_mag')

    # 1) Temperature scan
    temperatures = np.linspace(startTemp, endTemp, numberTemp)
    energies = np.zeros(numberTemp)
    magnetizations = np.zeros(numberTemp)

    for idx,T in enumerate(temperatures):
        beta = 1.0 / T
        print(f"Running T={T:.3f}, algorithm={algorithm} ({idx+1}/{numberTemp})")

        lattice = ising.init_lattice(N,M, init_type="random")
        if algorithm == "metropolis":
            E_avg, M_avg = ising.run_metropolis(lattice, beta, eqSteps, measureSteps)
        elif algorithm == "wolff":
            E_avg, M_avg = ising.run_wolff(lattice, beta, eqSteps, measureSteps)
        else:
            raise ValueError(f"Unknown algorithm {algorithm}")

        energies[idx] = E_avg
        magnetizations[idx] = M_avg / (N*M)

    os.makedirs(os.path.dirname(path_ene), exist_ok=True)
    np.save(path_ene, energies)
    np.save(path_mag, magnetizations)
    print("Finished T-scan: energies, magnetizations saved.")

    # 2) Time evolution at single T
    lattice_te = ising.init_lattice(N,M, "random")
    beta_te = 1.0 / timeEvolutionTemp
    steps_demo = 200
    time_evol = []
    for t in range(steps_demo):
        if algorithm == "metropolis":
            ising.metropolis_sweep(lattice_te, beta_te)
        else:
            ising.wolff_cluster_sweep(lattice_te, beta_te)
        time_evol.append( ising.calculate_magnetization(lattice_te)/(N*M) )

    time_evol = np.array(time_evol)
    np.save(path_time, time_evol)
    print("Time-evolution data saved.")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python simulation.py configuration.txt")
        sys.exit(1)

    main(sys.argv[1])
