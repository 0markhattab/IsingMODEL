# ising.py

import numpy as np
from collections import deque

def init_lattice(N, M, init_type="random"):
    if init_type == "random":
        lattice = np.random.choice([-1, 1], size=(N, M))
    elif init_type == "all_up":
        lattice = np.ones((N, M), dtype=int)
    elif init_type == "all_down":
        lattice = -1 * np.ones((N, M), dtype=int)
    else:
        raise ValueError("init_type must be 'random', 'all_up', or 'all_down'.")
    return lattice

def calculate_energy(lattice, J=1.0, h=0.0):
    N, M = lattice.shape
    E = 0.0
    for i in range(N):
        for j in range(M):
            s_ij = lattice[i, j]
            s_right = lattice[i, (j + 1) % M]
            s_down  = lattice[(i + 1) % N, j]
            E -= J * s_ij * s_right
            E -= J * s_ij * s_down
    # external field
    E_field = -h * np.sum(lattice)
    E += E_field
    return E

def calculate_magnetization(lattice):
    return np.sum(lattice)

# ---------------------------------------------------------------------
# Metropolis
# ---------------------------------------------------------------------
def metropolis_sweep(lattice, beta, J=1.0, h=0.0):
    N, M = lattice.shape
    for _ in range(N*M):
        i = np.random.randint(0, N)
        j = np.random.randint(0, M)

        s_old = lattice[i, j]
        top    = lattice[(i-1) % N, j]
        bottom = lattice[(i+1) % N, j]
        left   = lattice[i, (j-1) % M]
        right  = lattice[i, (j+1) % M]
        neighbor_sum = top + bottom + left + right

        # bonds
        delta_E_bond  = 2.0 * J * s_old * neighbor_sum
        # field
        delta_E_field = 2.0 * h * s_old
        delta_E = delta_E_bond + delta_E_field

        if delta_E <= 0:
            lattice[i, j] = -s_old
        else:
            if np.random.rand() < np.exp(-beta * delta_E):
                lattice[i, j] = -s_old

def run_metropolis(lattice, beta, eqSteps, measureSteps, J=1.0, h=0.0):
    for _ in range(eqSteps):
        metropolis_sweep(lattice, beta, J, h)

    E_accum = 0.0
    M_accum = 0.0
    for _ in range(measureSteps):
        metropolis_sweep(lattice, beta, J, h)
        E_accum += calculate_energy(lattice, J, h)
        M_accum += calculate_magnetization(lattice)

    E_avg = E_accum / measureSteps
    M_avg = M_accum / measureSteps
    return E_avg, M_avg

# ---------------------------------------------------------------------
# Wolff
# ---------------------------------------------------------------------
def _neighbors(site, N, M):
    i, j = site
    return [((i-1) % N, j), ((i+1) % N, j),
            (i, (j-1) % M), (i, (j+1) % M)]

def wolff_cluster_update(lattice, beta, J=1.0, h=0.0):
    N, M = lattice.shape
    i_seed = np.random.randint(0, N)
    j_seed = np.random.randint(0, M)
    seed_spin = lattice[i_seed, j_seed]

    p_add = 1.0 - np.exp(-2.0 * beta * J)

    from collections import deque
    cluster_sites = deque()
    cluster_sites.append((i_seed, j_seed))

    lattice[i_seed, j_seed] = -seed_spin
    cluster_size = 1

    while cluster_sites:
        site = cluster_sites.pop()
        for nbr in _neighbors(site, N, M):
            if lattice[nbr] == seed_spin:
                if np.random.rand() < p_add:
                    lattice[nbr] = -seed_spin
                    cluster_sites.appendleft(nbr)
                    cluster_size += 1
    return cluster_size

def wolff_cluster_sweep(lattice, beta, J=1.0, h=0.0):
    wolff_cluster_update(lattice, beta, J, h)

def run_wolff(lattice, beta, eqSteps, measureSteps, J=1.0, h=0.0):
    for _ in range(eqSteps):
        wolff_cluster_sweep(lattice, beta, J, h)

    E_accum = 0.0
    M_accum = 0.0
    for _ in range(measureSteps):
        wolff_cluster_sweep(lattice, beta, J, h)
        E_accum += calculate_energy(lattice, J, h)
        M_accum += calculate_magnetization(lattice)

    E_avg = E_accum / measureSteps
    M_avg = M_accum / measureSteps
    return E_avg, M_avg
