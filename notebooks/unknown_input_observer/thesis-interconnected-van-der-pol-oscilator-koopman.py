# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import control as ct
import matplotlib.pyplot as plt
import math
import numpy as np
import cvxpy as cp
import networkx as ntx
from scipy.integrate import solve_ivp
import os
import copy
import random
from control_utils import levants, uio
from IPython.display import clear_output

from itertools import product

from control_utils.politopic_representation import *

plt.style.use(['default', './style.mplstyle'])

# rng = np.random.default_rng()
rng_seed = 5

np.random.seed(rng_seed)
random.seed(rng_seed)

# %%
# constants
mi = np.array([1, .8, .6, .4, .2, .1, .05])
eps = 1e-9 #
eta = 1 # decay rate

# simulation time
time = 40

# normal cases
z_i_interval = [[0, 14.4]]
zeta_i_interval = [[0.064, 0.099]]

# hyperplanes
a_i = np.array([
    [1, -1, 0, 0],
    [0, 0, 1, -1]
])
b_i = np.array([12, 12, 12, 12]).reshape(4, 1)
bR_i = np.array([5, 5, 5, 5]).reshape(4, 1)

# Connections Graph construction
N = 7

E_half = [(1, 2), (1, 3), (1, 5), (1, 6), (3, 4), (4, 7), (5, 7)]
x0_systems = np.array([
     2,  -3,
     2,  1,
    -1,  0,
     0, -2,
     0, -.4,
    -2,  2,
     1, -1,
])

# x0_systems = np.array([
#      0, -.4,
#      0, -.4,
#      0, -.4,
#      0, -.4,
#      0, -.4,
#      0, -.4,
#     #  2,  1,
#     # -1,  0,
#     #  0, -2,
#     #  0, -.4,
#     # -2,  2,
# ])

for i in range(N):
    id = i*2
    print(math.dist((0, 0), x0_systems[id:id+2]))


# lambda_y1 = 2.5*np.array([
#     [42, 6, 2.2, 1.8, .8, .4],
#     [42, 6, 2.2, 1.8, .8, .4],
#     [42, 6, 2.2, 1.8, .8, .4],
#     [42, 6, 2.2, 1.8, .8, .4],
#     [42, 6, 2.2, 1.8, .8, .4],
#     [42, 6, 2.2, 1.8, .8, .4],
# ])

# lambda_y1 = 1.3*np.array([
#     [48, 35, 15, 4, 1.5, .3],
#     [48, 35, 15, 4, 1.5, .3],
#     [48, 35, 15, 4, 1.5, .3],
#     [48, 35, 15, 4, 1.5, .3],
#     [48, 35, 15, 4, 1.5, .3],
#     [48, 35, 15, 4, 1.5, .3],
# ])

lambda_y1 = 1.5*np.array([
    [30, 12, 12, 12, 6],
    [30, 12, 12, 12, 6],
    [30, 12, 12, 12, 6],
    [30, 12, 12, 12, 6],
    [30, 12, 12, 12, 6],
    [30, 12, 12, 12, 6],
    [30, 12, 12, 12, 6],
])

# ok
# lambda_y1 = 1*np.array([
#     [35, 20, 26, 12, 9, 0.7],
#     [35, 20, 26, 12, 9, 0.7],
#     [35, 20, 26, 12, 9, 0.7],
#     [35, 20, 26, 12, 9, 0.7],
#     [35, 20, 26, 12, 9, 0.7],
#     [35, 20, 26, 12, 9, 0.7],
# ])

nlevants = lambda_y1[0].shape[0]
x0_observer = np.zeros(x0_systems.shape)
x0_levants = np.zeros(nlevants*N)

# --- Koopman Observable Setup ---
K_koop = 6

def phi_koop(v):
    return np.array([
        np.sin(v),        # 1st Harmonic Sine
        np.cos(v),        # 1st Harmonic Cosine
        np.sin(2*v),      # 2nd Harmonic Sine
        np.cos(2*v),      # 2nd Harmonic Cosine
        np.sin(3*v),      # 3rd Harmonic Sine
        np.cos(3*v)       # 3rd Harmonic Cosine
    ], dtype=float)

n_edges = len(E_half)
x0_koopman = np.zeros(n_edges * K_koop)

x0 = np.concat([x0_systems, x0_observer, x0_levants, x0_koopman], axis=0)

G = ntx.Graph()
G.add_nodes_from(range(1, N+1))
G.add_edges_from(E_half)

dG = ntx.DiGraph(G)
dG.remove_edges_from(E_half)


# %%
def get_hybrid_schedule(N, E_half):
    adj = {i: [] for i in range(N)}
    deg = {i: 0 for i in range(N)}
    for k, (u, v) in enumerate(E_half):
        u -= 1
        v -= 1
        adj[u].append((v, k, 1))
        adj[v].append((u, k, -1))
        deg[u] += 1
        deg[v] += 1
        
    leaves = [i for i in range(N) if deg[i] == 1]
    schedule = []
    cycle_edges = set(range(len(E_half)))
    
    while leaves:
        u = leaves.pop(0)
        for v, k, direction in adj[u]:
            if k in cycle_edges:
                cycle_edges.remove(k)
                schedule.append((u, v, k, direction))
                deg[v] -= 1
                if deg[v] == 1:
                    leaves.append(v)
                break
                
    return schedule, list(cycle_edges)

tree_schedule, cycle_edges = get_hybrid_schedule(N, E_half)

def plot_G():
    plt.figure(dpi=150, constrained_layout=True, figsize=(4, 4))
    # plt.subplot(1, 2, 1)
    plt.title("Hybrid Topology Decomposition")
    nodes_pos = ntx.spring_layout(G, pos={'0': (0, 0)}, iterations=500, seed=rng_seed)
    
    ntx.draw_networkx_nodes(G, pos=nodes_pos, node_color='lightgray', edgecolors='k')
    ntx.draw_networkx_labels(G, pos=nodes_pos)
    
    tree_edges_list = [E_half[k] for _, _, k, _ in tree_schedule]
    cycle_edges_list = [E_half[k] for k in cycle_edges]
    
    ntx.draw_networkx_edges(G, pos=nodes_pos, edgelist=tree_edges_list, edge_color='blue', width=2)
    ntx.draw_networkx_edges(G, pos=nodes_pos, edgelist=cycle_edges_list, edge_color='red', width=2)
    
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='blue', linewidth=2, label='Algebraic (Tree)')
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='Koopman (Cycle)')
    plt.legend(handles=[blue_line, red_line], loc='lower left', fontsize=8)

    # plt.subplot(1, 2, 2)
    # plt.title("Directed Graph")
    # ntx.draw(dG, pos=nodes_pos, with_labels=True, node_color='lightgray', edgecolors='k')
    plt.show()

plot_G()
# %%
# kind of a constant in the way it is been built
a_cell = []
b_cell = []
bR_cell = []

for i in range(N):
    a_cell.append(a_i)
    b_cell.append(b_i)
    bR_cell.append(bR_i)

# %%
E_half = np.array(E_half)
if E_half.max() > N:
    raise Exception("Error in the connections set")

E_G = np.concat([E_half, np.flip(E_half, axis=1)], axis=0)
V_G = np.arange(start=1, stop=N+1)
A_G = np.zeros((N, N))
D_G = np.zeros((N, N))
L_G = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        is_in_array = np.any(np.all(E_G == [i+1, j+1], axis=1))

        if is_in_array:
            A_G[i, j] = 1

for i in range(N):
    D_G[i, i] = np.sum(A_G[i,:])

L_G = D_G - A_G

# %%
# generate combinations for multi-index -> j, k \in \mathbb{I}_{n_z} and l \in \mathbb{I}_{n_zeta}
n_z = len(z_i_interval)
n_zeta = len(zeta_i_interval)

interval_size = 2 # Just to make this information explict

# TODO: FIX THESE PERMUTATIONS
j_perms, _ = permn(np.arange(interval_size, dtype=int), n_z)
k_perms, _ = permn(np.arange(interval_size, dtype=int), n_zeta)
l_perms, _ = permn(np.arange(interval_size, dtype=int), n_z)


# %%
def A_ijl(z: list[np.ndarray], zeta: list[np.ndarray], **kwargs) -> np.ndarray:
    return np.array([
        [0, 10 + z[0]],
        [0, 0],
    ])

def C_ijl(z: list[np.ndarray], zeta: list[np.ndarray], **kwargs) -> np.ndarray:
    return np.array([
        [zeta[0], 0]
    ]).reshape((1, 2))
# %%
A_cell = []
C_cell = []

for i in range(N):
    A_i_cell = []
    C_i_cell = []

    for j_p, k_p in product(j_perms, l_perms):

        jk_perm = np.concatenate((j_p, k_p), dtype=int)

        A_ind = bin_perm_to_dec(jk_perm)
        C_ind = A_ind

        z = []
        for j in range(n_z):
            z.append(z_i_interval[j][jk_perm[j]])
        
        zeta = []
        for j in range(n_zeta):
            zeta.append(zeta_i_interval[j][jk_perm[n_z + j]])

        A_i_cell.append(A_ijl(z, zeta))
        C_i_cell.append(C_ijl(z, zeta))
    
    A_cell.append(A_i_cell)
    C_cell.append(C_i_cell)

# %%
j_perms_plus = perm_plus(j_perms)
k_perms_plus = perm_plus(k_perms)
l_perms_plus = perm_plus(l_perms)

nx = []
ny = []

for i in range(N):
    nx_i = A_cell[i][0].shape[0]
    ny_i = C_cell[i][0].shape[0]
    
    nx.append(nx_i)
    ny.append(ny_i)

P_trace = 0
P_cell = []
L_till_cell = []

for i in range(N):
    P_i = cp.Variable((nx[i], nx[i]), symmetric=True)
    P_cell.append(P_i)

    L_till_i_cell = []
    for perm in l_perms:
        L_till_i = cp.Variable((nx[i], ny[i]))
        L_till_i_cell.append(L_till_i)
    
    L_till_cell.append(L_till_i_cell)

constrains = []

for i in range(N):

    P_trace += cp.trace(P_cell[i])
    constrains += [P_cell[i] >> eps] # the same as P > 0

    for m_index, n_index, o_index in product(j_perms_plus, k_perms_plus, l_perms_plus):
        # for n_index in k_perms_plus:

        # TODO: fix this below
        m_perms = multi_index_permutation(copy.deepcopy(m_index))
        n_perms = multi_index_permutation(copy.deepcopy(n_index))
        o_perms = multi_index_permutation(copy.deepcopy(o_index))

        Upsilon_mno_sum = 0
        for m_perm, n_perm, o_perm in product(m_perms, n_perms, o_perms):
        # for m_perm in m_perms:
            # for n_perm in n_perms:
            # jl_perm = m_perm
            mn_perm = np.concatenate((m_perm, n_perm), dtype=int)
            
            A_ind = bin_perm_to_dec(copy.deepcopy(mn_perm))
            L_ind = bin_perm_to_dec(copy.deepcopy(o_perm))
            C_ind = A_ind

            M = P_cell[i]@A_cell[i][A_ind] - L_till_cell[i][L_ind]@C_cell[i][C_ind] + eta*P_cell[i]
            Upsilon_mno_sum += M + M.T
    
        if Upsilon_mno_sum:
            constrains += [Upsilon_mno_sum << -eps]
        
    for j in range(a_cell[i].T.shape[0]):
        aj = np.expand_dims(a_cell[i].T[j], axis=1)
        phi = uio.phi_z(a_cell[i].T, aj, bR_cell[i])
        sj = 1/(b_cell[i][j] - phi) * aj

        M = cp.bmat([
            [np.eye(1), sj.T],
            [sj       , P_cell[i]],
        ])
        constrains += [ M >> 0 ] # >= breaks -> i know why but cannot remember right now


prob = cp.Problem(cp.Minimize(P_trace), constraints=constrains)
result = prob.solve(solver=cp.MOSEK, verbose=True)

# %%
is_positive_defined = []
for i in range(N):
    i_is_positive_defined = True if all(np.linalg.eig(P_cell[i].value).eigenvalues > 0) else False
print(f'e.g.: P[1] = \n{P_cell[0].value}')
print(f"The solution is \'{prob.status}\', with Trace(P) = {result:.2f} and P is {"Positive Defined" if all(is_positive_defined) else "NOT Positive Defined"}")

# %%
L_cell = []

for i in range(N):
    P_i_inv = np.linalg.inv(P_cell[i].value)

    L_till = []
    for j in range(len(L_till_cell[i])):
        L_till.append(P_i_inv@L_till_cell[i][j].value)
    L_cell.append(L_till)

# %%
L_cell[0]

# %%
# simulação

def f_x(x_i, mi):
    return np.array([
        [x_i[1]],
        [-x_i[0] + mi*(1 - x_i[0]**2)*x_i[1]],
    ])

def f_z(z_i, mi):
    return np.array([
        [z_i[1]*(z_i[0]**2 + 100) / 10],
        [-10*z_i[0] / (z_i[0]**2 + 100) + mi*(1 - z_i[0]**2)*z_i[1] - z_i[0]*z_i[1]**2 / 5]
    ])


def gd_x(i: int, n: int, N: int, x: np.ndarray, G: ntx.Graph) -> list[np.ndarray[float], np.ndarray[float]]: # check if this typing is correct
    """Calculates UIs

    Parameters
    ----------
    i : int
        Current system - 1 based index
    n : int
        Number of states
    N : int
        Number of subsystems
    x : np.ndarray
        All states
    G : ntx.Graph
        Graph of interconnections
    
    Returns
    ----------
    G*d(x), d(x) : np.ndarray[float], np.ndarray
    """

    gd = np.zeros(n)
    gd_decoupled = []
    
    i_id = (i-1) * n
    x_i = x[i_id:i_id+n]

    for edge in G.edges(i):
        j = edge[1] - 1 # mapping from 1~N to 0~N-1
        j_id = j * n
        x_j = x[j_id:j_id+n]

        # affects only the second state
        # print(f"x_j[{j_id/2}]: {x_j}, x_i[{i-1}]: {x_i}")
        d = -2*math.sin(x_j[1] - x_i[1])
        gd[1] += d

        # print(f"edge: {edge}, d: {d}")

        gd_decoupled.append(d)

    gd = gd.reshape((n, 1))
    gd_decoupled = np.array(gd_decoupled).reshape(len(gd_decoupled), 1)

    return gd, gd_decoupled

def h_x(x_i):
    return np.array([math.atan(x_i[0]/10)])

def G_i(z: np.ndarray) -> np.ndarray: # TODO: FIX FOR Z -> test
    return np.array([
        [0],
        [-10/(z[0]**2 + 100)]
    ]).reshape((2, 1))

def Gamma_inv(z: np.ndarray) -> np.ndarray: # TODO: FIX FOR Z -> test - G_i may have fixed this
    """Calculates Gamma_inv(x)

    Returns
    ----------
    Gamma_inv : np.ndarray
    """
    Gamma = G_i(z)[1]

    return np.array(1/Gamma)

def Q_z(z: np.ndarray) -> np.ndarray: # TODO: FIX FOR Z -> test - this should result to [0 ; 1]
    """Calculates Q(x)
    
    Returns
    ----------
    Q(x) : np.ndarray
    """
    g_i = G_i(z)
    gamma_inv = Gamma_inv(z)
    Q  = g_i@gamma_inv #

    return np.expand_dims(Q, axis=1)

def Psi_x(x_i: np.ndarray, mi):
    Psi = f_x(x_i, mi)[1]
    Psi = np.array(Psi).reshape((len(Psi), 1))

    return Psi

def Psi_z(x_i: np.ndarray, mi):
    Psi = f_z(x_i, mi)[1]
    Psi = np.array(Psi).reshape((len(Psi), 1))

    return Psi

def dot_Y_levants(lambda_y1: np.ndarray[float], y: np.ndarray[float], Y_levants: np.ndarray[float]) -> np.ndarray[float]:
    """Returns dot_Y_levants"""

    # can be adapted for multiple signals
    # z1 = y1 = x2
    # z2 = dot_y1 = dot_x2
    # Only necessary up to z2 but higher order helps improve quality 

    n_y1 = lambda_y1.shape[0]
    id_y1 = 0

    z_y1 = Y_levants[id_y1:n_y1]
    y1 = y[0]

    dot_y1 = levants.differentiator(y1, z_y1, lambda_y1)

    return np.concat([dot_y1], axis=0)

def model(t: float, x: np.ndarray[float], nx: int, G: ntx.Graph, mi: float, E_half: list, _progress=[0.0]):
    if t - _progress[0] >= 0.2:
        print(f"Simulating t={t:.1f}s", end='\r')
        _progress[0] = t

    xdot = np.zeros(shape=x.shape)
    xdot = np.expand_dims(xdot, axis=1)

    N = G.number_of_nodes()

    d_hat_all = np.zeros(N)
    
    # Store translated x_hat for RBF network
    x_hat_real_all = []
    
    for i in range(N):
        id = i * nx
        id_hat = id + N*nx
        id_Y_levants = i*nlevants + 2*N*nx

        x_i = x[id:id+nx]
        x_hat_i = x[id_hat:id_hat+nx] # this is actually z_hat
        Y_levants_i = x[id_Y_levants:id_Y_levants+nlevants]
        
        # Translate z_hat to x_hat
        x_hat_1_real = x_hat_i[0]
        x_hat_2_real = x_hat_i[1] * (x_hat_i[0]**2 + 100) / 10
        x_hat_real_all.append(np.array([x_hat_1_real, x_hat_2_real]))
        
        i_mapped = i + 1
        d_i, d_i_decoupled = gd_x(i_mapped, nx, N, x, G)
        
        alpha_i = [np.array([x_hat_i[0]**2/10])]

        xdot[id:id+nx] = f_x(x_i, mi[i]) + (10/(100 + x_i[0]**2))*d_i
        y_i = h_x(x_i)
        y_i_hat = h_x(x_hat_i)

        Q = Q_z(x_hat_i)
        Psi_hat = Psi_z(x_hat_i, mi[i])

        dot2 = Y_levants_i[2]*10
        dot1 = Y_levants_i[1]*10
        Y = np.array([
            Y_levants_i[2]
        ])

        L = L_alpha(alpha_i, z_i_interval, l_perms, L_cell[i])

        d_hat_decoupled = Gamma_inv(x_hat_i)@(Y - Psi_hat)*10 
        d_hat_all[i] = float(d_hat_decoupled)

        delta = np.expand_dims(y_i - y_i_hat, axis=1)
        xdot[id_hat:id_hat+nx] = f_z(x_hat_i, mi[i]) + Q@(Y - Psi_hat) + L@delta

        dot_Y = dot_Y_levants(lambda_y1[i], h_x(x_i), Y_levants_i)
        xdot[id_Y_levants:id_Y_levants+nlevants] = dot_Y

    # --- DUIO Koopman Observable Update (Hybrid) ---
    n_edges = len(E_half)
    koop_start_id = 2*N*nx + N*nlevants
    C_koop = x[koop_start_id:].reshape(n_edges, K_koop)
    C_koop_dot = np.zeros_like(C_koop)
    
    # 1. Algebraic Reconstruction for Trees
    D_res = np.copy(d_hat_all)
    for u, v, k, direction in tree_schedule:
        D_res[v] += D_res[u]
        D_res[u] = 0
        
    # 2. Koopman for Cycles
    d_NN_all = np.zeros(N)
    phi_edges_k_map = {}
    
    for k in cycle_edges:
        edge = E_half[k]
        i = edge[0] - 1
        j = edge[1] - 1
        
        delta_x = x_hat_real_all[j][1] - x_hat_real_all[i][1]
        phi_k = phi_koop(delta_x)
        phi_edges_k_map[k] = phi_k
        
        g_hat_ij = np.dot(C_koop[k], phi_k)
        
        d_NN_all[i] += g_hat_ij
        d_NN_all[j] -= g_hat_ij
        
    e = np.clip(D_res - d_NN_all, -20.0, 20.0)
    gamma_lr = 10.0 if t > 0.5 else 0.0
    C_max = 5.0   # safe bound: true sin(v) coefficient is ~2.0, so this is well above it
    sigma = 2.0   # sigma modification gain
    
    for k in cycle_edges:
        edge = E_half[k]
        i = edge[0] - 1
        j = edge[1] - 1
        phi_k = phi_edges_k_map[k]
        # Normalized gradient: divides by (1 + phi^T phi) to bound the step size and prevent overshoot
        norm_factor = 1.0 + phi_k @ phi_k
        norm_C = np.linalg.norm(C_koop[k])
        sigma_mod = sigma * max(0.0, norm_C - C_max) / (norm_C + 1e-9) if norm_C > C_max else 0.0
        C_koop_dot[k] = gamma_lr * (e[i] - e[j]) * phi_k / norm_factor - sigma_mod * C_koop[k]
        
    xdot[koop_start_id:] = np.expand_dims(C_koop_dot.flatten(), axis=1)

    return xdot.flatten()


sim_time = (0, time)

result = solve_ivp(model, sim_time, x0, args=(A_cell[0][0].shape[0], G, mi, E_half), method="RK45", dense_output=False)
clear_output(wait=False)

t = result.t
x = result.y
print(f"Simulation finished: Reached t={t[-1]:.2f}s with {len(t)} steps. Success: {result.success}")

# --- Post-simulation dist_hist reconstruction ---
print("Reconstructing histories...")
dist_hist = [t]
for i in range(N):
    dist_hist.append([[], [], []])

nx_val = A_cell[0][0].shape[0]
for m in range(len(t)):
    x_val = x[:, m]
    for i in range(N):
        id = i * nx_val
        id_hat = id + N*nx_val
        id_levants = i*nlevants + 2*N*nx_val
        
        x_i = x_val[id:id+nx_val]
        x_hat_i = x_val[id_hat:id_hat+nx_val]
        Y_levants_i = x_val[id_levants:id_levants+nlevants]
        
        i_mapped = i + 1
        d_i, d_i_decoupled = gd_x(i_mapped, nx_val, N, x_val, G)
        dist_hist[i+1][0].append(d_i_decoupled)
        
        Psi_hat = Psi_z(x_hat_i, mi[i])
        Y_val = np.array([Y_levants_i[2]])
        d_hat_decoupled = Gamma_inv(x_hat_i)@(Y_val - Psi_hat)*10 
        dist_hist[i+1][1].append([float(d_hat_decoupled)])
        
        xdot_true = f_x(x_i, mi[i]) + (10/(100 + x_i[0]**2))*d_i
        dist_hist[i+1][2].append(xdot_true)

for i in range(1, len(dist_hist)):
    for j in range(len(dist_hist[i])):
        dist_hist[i][j] = np.array(dist_hist[i][j])
print("Done!")

# %%
# Translate from z_hat -> x_hat # TODO: for some reason it is affecting the wrong things
for i in range(N):
    id = i * nx[0]
    id_hat = id + N*nx[0]

    for t_i in range(len(t)):
        x[id_hat + 1][t_i] = x[id_hat + 1][t_i] * (x[id_hat][t_i]**2 + 100) / 10

# %%
# plt.figure()
# plt.grid()
# plt.plot(dist_hist[0], np.sum(dist_hist[1][0], axis=1), label='dist x1')


# %%

# Levants differentiator tuning
i = 0 # system
id = i * nx[0]

# h_vectorized = np.vectorize(h_x)

# OK
ii = 0 # max = 2 # state
id_Y_levants = i * nlevants + 2*N*nx[0] + ii
plt.figure()
plt.title("levant y0=l0")
plt.plot(t, np.apply_along_axis(h_x, 0, x[id:id+nx[0]]).flatten(), 'b', label=f'y0') # only for id = 0 or 1
plt.plot(t, x[id_Y_levants], 'r--', label='l0')
plt.legend()
plt.ylim([-.5, .5])
plt.show()

# NOK
# TODO: WHY I HAVE TO USE THE 10x
ii = 1
id_Y_levants = i * nlevants + 2*N*nx[0] + ii
plt.figure()
plt.title("levant y0'=l1")
plt.plot(t, np.apply_along_axis(f_x, 0, x[id:id+nx[0]], mi=mi[i])[0, :, :].flatten(), 'b', label="y0'") # only for id = 0 or 1 -> TODO: fix for dot z1 equation
plt.plot(t, x[id_Y_levants]*10, 'r--', label='l1')
plt.legend()
# plt.ylim([-5, 5])
plt.show()

# IN THEORY OK
# TODO: WHY I HAVE TO USE THE 10x
ii = 2
id_Y_levants = i * nlevants + 2*N*nx[0] + ii
plt.figure()
plt.title("levant y0''=l2")
plt.plot(t, dist_hist[i+1][2][:, 1], 'k' , label="y0''") # only for id = 0 or 1 -> TODO: fix for dot z2 equation
plt.plot(t, x[id_Y_levants]*10, 'r--', label='l2')
plt.legend()
# plt.ylim([-5, 5])
plt.show()

id = i * nx[0]
id_Y_levants = i * nlevants + 2*N*nx[0] + 2
Y = x[id_Y_levants]

result = []
psi_hist = []
for ii in range(len(t)):
    result.append(Gamma_inv(x[id:id+2][:, ii])@(Y[i] - Psi_z(x[id:id+2][:, ii], mi[i])))
    psi_hist.append(Psi_z(x[id:id+2][:, ii], mi[i])[0])

plt.figure()
plt.title('levant with rebuilt signals')
plt.plot(t, np.array(result), 'k', label='d(x)')
plt.plot(t, Y, 'r--', label='Y(hat x)')
plt.plot(t, np.array(psi_hist).flatten() + np.sum(dist_hist[i+1][1], axis=1), 'g--', label='Z(x)')
plt.ylim([-5, 5])
plt.legend()
plt.show()

# # %%
# plt.figure()
# i = 1
# plt.plot(t, dist_hist[1][0][:, 0])

# %%
# %%
# Reconstruct edge signals from the Koopman coefficients
koop_start_id = 2*N*nx[0] + N*nlevants
n_edges = len(E_half)
g_hat_all = np.zeros((n_edges, len(t)))

for m in range(len(t)):
    # 1. Algebraic Reconstruction for Trees
    D_res = np.array([float(dist_hist[i+1][1][m][0]) for i in range(N)])
    
    for u, v, k, direction in tree_schedule:
        if direction == 1:
            g_hat_all[k, m] = D_res[u]
        else:
            g_hat_all[k, m] = -D_res[u]
        
        D_res[v] += D_res[u]
        D_res[u] = 0
        
    # 2. Koopman Operator for Cycles
    for k in cycle_edges:
        edge = E_half[k]
        i = edge[0] - 1
        j = edge[1] - 1
        
        i_hat_id = i * nx[0] + N * nx[0] + 1
        j_hat_id = j * nx[0] + N * nx[0] + 1
        
        delta_x = x[j_hat_id, m] - x[i_hat_id, m]
        
        C_k = x[koop_start_id + k*K_koop : koop_start_id + (k+1)*K_koop, m]
        phi_k = phi_koop(delta_x)
        g_hat_all[k, m] = np.dot(C_k, phi_k)

# %%
def plot_graph(x, nx):
    plt.figure(dpi=150, constrained_layout=True)
    # plt.suptitle(f'{N}-Interconnected Oscilators', y=1.0)

    cols = 3
    rows = math.ceil(N/cols)

    for i in range(N):
        ind = i * nx
        ind_hat = ind + N*nx

        plt.subplot(rows, cols, i+1)
        plt.title(f"Oscillator {i+1}", fontsize=8)
        plt.scatter(x0[ind], x0[ind+1], s=30, facecolors='none', edgecolors='k') #, label=f'$\\mathbf{{x}}_{i+1}$')
        plt.plot(x[ind, :], x[ind+1, :], 'k') #, label=f'$\\mathbf{{x}}_{i+1}$')
        plt.plot(x[ind_hat, :], x[ind_hat+1, :], 'r--', linewidth=1) #, label=f'$\\mathbf{{\\hat{{x}}}}_{i+1}$')
        # plt.xlabel("$x_1$")
        # plt.ylabel("$x_2$")
        # plt.legend()
        
        plt.gca().set_box_aspect(1)
        plt.grid(which='both')
    
    legend = plt.figlegend([r'$\mathbf{x}_{i0}$', r'$\mathbf{x}_i$', r'$\mathbf{\hat{x}}_i$'], loc = 'lower center', ncol = 3,  bbox_to_anchor=(0.5, -0.1))

    dir = os.path.join(os.curdir, f'.figures/')
    if not os.path.isdir(dir):
        os.mkdir(dir)

    plt.savefig(f".figures/non_y_dynamics_{N}", dpi=900, bbox_inches='tight', bbox_extra_artists=[legend])
    plt.show()

def plot_one_graph(i):
    # i = 3
    id = (i - 1)*nx[0]
    id_hat = id + N*nx[0]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.margins(x=0)
    plt.plot(t, x[id, :], 'k', label=f'$x_{{{i}1}}$')
    plt.plot(t, x[id_hat, :], 'r--', label=f'$\\hat{{x}}_{{{i}1}}$')
    # plt.ylim([-.02, .02])
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.margins(x=0)
    plt.plot(t, x[id+1, :], 'k',  label=f'$x_{{{i}2}}$')
    plt.plot(t, x[id_hat+1, :], 'r--',  label=f'$\\hat{{x}}_{{{i}2}}$')
    # plt.ylim([-.08, .08])
    plt.xlabel('$t \\; (s)$')
    plt.grid()
    plt.legend()
    plt.savefig(f".figures/non_y_single_dynamics_{i}", dpi=900, bbox_inches='tight')#, bbox_extra_artists=[legend])
    plt.show()

    # # Error dynamics of one oscillator
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.margins(x=0)
    # plt.plot(t, x[id, :] - x[id_hat, :], 'k', label=f'$e_{{{i}1}}$')
    # plt.plot(t, np.ones(t.shape)*6, 'r--', label='bounds of $\mathcal{E}$')
    # plt.plot(t, np.ones(t.shape)*-6, 'r--')
    # # plt.plot(t, x[id_hat, :], 'r--', label=f'$x_{{{i}1}}$')
    # plt.xlim([0, 2.5])
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.subplot(2, 1, 2)
    # plt.margins(x=0)
    # plt.plot(t, x[id+1, :] - x[id_hat+1, :], 'k',  label=f'$e_{{{i}2}}$')
    # plt.plot(t, np.ones(t.shape)*6, 'r--', label="bounds of $\mathcal{E}$")
    # plt.plot(t, np.ones(t.shape)*-6, 'r--')
    # # plt.plot(t, x[id_hat+1, :], 'r--',  label=f'$x_{{{i}2}}$')
    # plt.xlim([0, 2.5])
    # plt.xlabel("$t \\; (s)$")
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.savefig(f".figures/error_dynamicssingle_dynamics_{i}", dpi=900, bbox_inches='tight')#, bbox_extra_artists=[legend])
    # plt.show()

def plot_error():
    cols = 2
    rows = math.ceil(N/cols)
    
    fig, axs = plt.subplots(rows, cols, dpi=150, constrained_layout=True, sharex=True, layout='constrained')

    for i, j in product(range(rows), range(cols)):
        ind_plot = i * cols + j
        
        if ind_plot >= N:
            axs[i, j].axis('off')
            continue

        ind = ind_plot * nx[0]
        ind_hat = ind + N*nx[0]

        # plt.subplot(rows, cols, i+1)
        axs[i, j].margins(x=0)
        axs[i, j].set_title(f"Oscillator {ind_plot+1}", fontsize=8)

        # axs[i, j].plot(t, np.sum(dist_hist[ind_plot+1][0], axis=1), 'k') #, label=f'$\\mathbf{{x}}_{i+1}$')
        # axs[i, j].plot(t, -np.sum(dist_hist[ind_plot+1][1], axis=1), 'r--', linewidth=1) #, label=f'$\\mathbf{{\\hat{{x}}}}_{i+1}$')
        
        axs[i, j].plot(t, x[ind, :] - x[ind_hat, :], 'k', label=f'$e_{{{i}1}}$')
        axs[i, j].plot(t, x[ind+1, :] - x[ind_hat+1, :], 'b',  label=f'$e_{{{i}2}}$')
        
        axs[i, j].plot(t, np.ones(t.shape)*7, 'r--', label='bounds of $\mathcal{E}$')
        axs[i, j].plot(t, np.ones(t.shape)*-7, 'r--')

        # plt.xlim([0, 2.5])
        # plt.xlabel("$x_1$")
        # plt.ylabel("$x_2$")
        # plt.legend()
        
        plt.grid()
        axs[i, j].set_xlim([0, 2.5])
        axs[i, j].set_ylim([-8, 8])
        axs[i, j].set_yticks([-7, -3.5, 0, 3.5, 7])
        axs[i, j].grid(which='both')
    
    legend = plt.figlegend([r'$e_{i1}$', r'$e_{i2}$', r'bounds of $\mathcal{E}$'], loc = 'lower center', ncol = 3,  bbox_to_anchor=(0.5, -0.1))

    # # dir = os.path.join(os.curdir, f'.figures/')
    # # if not os.path.isdir(dir):
    # #     os.mkdir(dir)

    fig.supxlabel("$t \\; (s)$")
    plt.savefig(f".figures/error_dynamics_{N}", dpi=600, bbox_inches='tight', bbox_extra_artists=[legend])
    plt.show()

def plot_graph_dist(dist_hist):
    # plt.figure(dpi=150, constrained_layout=True)

    cols = 2
    rows = math.ceil(N/cols)
    
    fig, axs = plt.subplots(rows, cols, dpi=150, constrained_layout=True, sharex=True, layout='constrained')

    for i, j in product(range(rows), range(cols)):
        ind_plot = i * cols + j
        
        if ind_plot >= N:
            axs[i, j].axis('off')
            continue

        ind = ind_plot * nx
        ind_hat = ind + N*nx

        # plt.subplot(rows, cols, i+1)
        axs[i, j].margins(x=0)
        axs[i, j].set_title(f"Oscillator {ind_plot+1}", fontsize=8)

        axs[i, j].plot(t, np.sum(dist_hist[ind_plot+1][0], axis=1), 'k') #, label=f'$\\mathbf{{x}}_{i+1}$')
        axs[i, j].plot(t, -np.sum(dist_hist[ind_plot+1][1], axis=1), 'r--', linewidth=1) #, label=f'$\\mathbf{{\\hat{{x}}}}_{i+1}$')
        
        # plt.xlabel("$x_1$")
        # plt.ylabel("$x_2$")
        # plt.legend()
        
        axs[i, j].set_ylim([-8.5, 8.5])
        axs[i, j].grid(which='both')
    
    legend = plt.figlegend([r'$\mathbf{d}_i$', r'$\mathbf{\hat{d}}_i$'], loc = 'lower center', ncol = 3,  bbox_to_anchor=(0.5, -0.1))

    # # dir = os.path.join(os.curdir, f'.figures/')
    # # if not os.path.isdir(dir):
    # #     os.mkdir(dir)

    fig.supxlabel("$t \\; (s)$")
    plt.savefig(f".figures/non_y_d_dynamics_{N}", dpi=600, bbox_inches='tight', bbox_extra_artists=[legend])
    plt.show()

def plot_graph_dist_ind(i, dist_hist, g_hat_all, G, E_half):
    plt.figure(dpi=150, constrained_layout=True)
    plt.margins(x=0)

    color = ['k', 'b', 'r', 'g', 'c']
    hat_color = ['k--', 'b--', 'r--', 'g--', 'c--']
    
    for j, edge in zip(range(dist_hist[i+1][0][0].shape[0]), G.edges(i+1)):
        plt.plot(t, dist_hist[i+1][0][:, j], color[j], label=f'$d_{{{edge[0]}{edge[1]}}}$')
        
        for k, e_half in enumerate(E_half):
            e_tuple = tuple(e_half)
            if e_tuple == edge:
                plt.plot(t, -g_hat_all[k, :], hat_color[j], linewidth=1, label=f'$\\hat{{d}}_{{{edge[0]}{edge[1]}}}$')
                break
            elif e_tuple == (edge[1], edge[0]):
                plt.plot(t, g_hat_all[k, :], hat_color[j], linewidth=1, label=f'$\\hat{{d}}_{{{edge[0]}{edge[1]}}}$')
                break
    
    plt.xlabel("$t \\; (s)$")
    plt.ylim([-6, 6])
    # plt.xlim([0, 10])
    plt.legend(loc=4, ncols=2)
    plt.grid(which='both')
    plt.savefig(f".figures/nn_d_rebuilt_i_{i+1}", dpi=900, bbox_inches='tight')
    plt.show()


# %%
dG.edges(3)

# %%
plot_graph(x, 2)

# %%
# Dynamics of one oscilator
plot_one_graph(3)

# %%
plot_error()

# %%
# Plot of dist sums and its estimation
plot_graph_dist(dist_hist)

# %%
# Plot of each individual dist and its individual reconstruction
id = 3
plot_graph_dist_ind(id, dist_hist, g_hat_all, G, E_half)

# %%
