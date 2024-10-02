import control as ct
import numpy as np
import matplotlib.pyplot as plt
from utils.set_operations import *
from h_inf_observer import h_inf_observer

### Autonomous ground vehicle

order = 10

# Constants
M   = 1476  # [Kg]
l_f = 1.13  # [m]
l_r = 1.49  # [m]
I_e = 442.8 # [kgm^2]
I_z = 1810  # [kgm^2]
C_f = 57000 # [N/rad]
C_r = 59000 # [N/rad]
C_x = 0.35
C_y = 0.45

v_x = [5, 30]       # [m/s]
v_y = [-3, 3]   # [m/s]
r_interval = [-2, 2] # [rad/s]

Ts = 0.01 # [s]

### Continuous Time Matrices
A_xi = lambda V_x, r: np.array([
    [0, r, 0],
    [0, -2*(C_f + C_r)/(M*(1/V_x)), 0],
    [0, 2*(l_r*C_r - C_f*l_f)/(I_z*(1/V_x)), 0]
])

f_v_xi = lambda V_x, r: np.array([
    -C_x*(1/V_x)**2/I_e,
    2*(C_r*l_r - C_f*l_f)*r/(M*(1/V_x)) - (1/V_x)*r,
    -2*(C_f*l_f**2 + C_r*l_r**2)*r/(I_z*(1/V_x))
]).reshape(3, 1)

V_x_interval = np.sort(1/np.array(v_x))
A_cell = []

for V_x_i in V_x_interval:
    for r_i in r_interval:
        A_cell.append(A_xi(V_x_i, r_i))

E_v = np.array([
    [1/I_e, 0],
    [0, 2*C_f/M],
    [0, 2*l_f*C_f/I_z]
])

E_vv = Ts * np.eye(2, 2) * 0

C = np.array([
    [1, 0, 0],
    [0, 0, 1]
])

G_v = np.array([0, -C_y/M, 0]).reshape(3, 1)

E_w = np.array([1, 10, 10]).reshape(3, 1)

#### Discrete Time Matrices
for i in range(len(A_cell)):
    A_cell[i] = Ts * A_cell[i] + np.eye(A_cell[i].shape[0])
    # f_v_cell[i] = Ts * f_v_cell[i]
f_v_xi_d = lambda V_x, r: Ts*f_v_xi(V_x, r)
E_d = Ts * E_v
G = Ts * G_v
E_w = Ts * E_w

def h_xi(V_x, r):

    omega_v1 = (V_x_interval[1] - V_x)/(V_x_interval[1] - V_x_interval[0])
    omega_v2 = (V_x - V_x_interval[0])/(V_x_interval[1] - V_x_interval[0])
    omega_r1 = (r_interval[1] - r)/(r_interval[1] - r_interval[0])
    omega_r2 = (r - r_interval[0])/(r_interval[1] - r_interval[0])

    h = [
        omega_v1*omega_r1, 
        omega_v1*omega_r2, 
        omega_v2*omega_r1, 
        omega_v2*omega_r2
    ]

    return h

def retrieve_R_theta(A_hat, G_h, L_h, E_w, E_v, R_till, R_theta, R_w, R_v):
    # compute R_till
    # R in this is R_till <---------------
    R_till_plus = np.concat([A_hat@R_till, G_h@R_theta, E_w@R_w, -L_h@E_v@R_v], axis=1) # use k=0 to calculate k=1

    # compute the boundaries of x_till
    x_high = +calculate_rs(R_till_plus) # check the number of intervals
    x_low = -calculate_rs(R_till_plus)  # check the number of intervals
    
    R_till_plus = reduce_zonotope(R_till_plus, order)

    x_till_interval = build_interval_from_bounds(x_low, x_high)

    phi_interval = np.array([
        [[0, 0], [2*v_y[0], 2*v_y[1]], [0, 0]]
    ])

    R_theta_interval = interval_product(phi_interval, x_till_interval)
    R_k_theta = zonotope_inclusion(R_theta_interval)

    return [R_till_plus, R_k_theta]

def vehicle_simulation(k, u, w, v, x, c, R, R_theta, R_till, A_cell, E_d, E_v, C, G, E_w, R_w, R_v, N = None, M = None):
    # membership update
    h = h_xi(1/x[0], x[2])
    f_xi = f_v_xi_d(1/x[0], x[2])
    A_h = 0
    G_h = 0

    phi = lambda p: p[1]**2
    
    # Model Simulation
    x_plus = 0
    for i in range(len(h)):
        x_plus += (h[i]*A_cell[i]@x).reshape(3,1)
        A_h += h[i] * A_cell[i]
        G_h += h[i] * G
    
    x_plus += E_d@u + f_xi + G*phi(x) + E_w*w
    y = C@x + E_v@v
    # --------------------------------------

    if N is None and M is None:
        # Criterion based estimation
        P_till = R@R.T
        omega = C@P_till@C.T + E_v@R_v@R_v.T@E_v.T
        Psi_h = A_h@P_till@C.T
        L_h = Psi_h@np.linalg.inv(omega)
    else:
        N_h = 0
        M_h = 0
        for i in range(len(h)):
            N_h += h[i] * N[i]
            M_h += h[i] * M[i]
        L_h = np.linalg.inv(N_h)@M_h

    # State estimation
    A_hat = A_h - L_h@C
    [R_till_plus, R_theta] = retrieve_R_theta(A_hat, G_h, L_h, E_w, E_v, R_till, R_theta, R_w, R_v)

    f_xi = f_v_xi_d(1/y[0], y[1])
    c_plus = A_hat@c + E_d@u + f_xi + G_h*phi(c) + L_h@y
    R_plus = np.concat([A_hat@R, G_h@R_theta, E_w@R_w, -L_h@E_v@R_v], axis=1)

    R_plus = reduce_zonotope(R_plus, order)

    return [x_plus, c_plus, R_plus, R_theta, R_till_plus, L_h]

### Simulation Variables
np.random.seed(2109)

nk = 30
w_max = 3
v1_max = 3
v2_max = .5
x_0 = np.array([5, -1, 0]).reshape(3, 1)
c_0 = x_0
R = np.array([
    [[-.2, .2]],
    [[-.5, .5]],
    [[-.5, .5]]
])
R_theta = np.array([[-1e6, 1e6]]).reshape(1, 1, 2)
R_till = R*1e3
# --------------------------------------------------

k = np.arange(0, Ts*nk, Ts)

torque = np.concat([
    5*np.ones(shape=(k.shape[0]//2, 1)),
    10*np.ones(shape=(k.shape[0]//2 + k.shape[0]%2, 1))
])
angle = np.concat([
    np.zeros(shape=(k.shape[0]//5, 1)),
    -.85*np.ones(shape=(k.shape[0]//5, 1)),
    np.zeros(shape=(k.shape[0]//5, 1)),
    1.2*np.ones(shape=(k.shape[0]//5 + k.shape[0]%5, 1)),
    np.zeros(shape=(k.shape[0]//5, 1))
])
u = np.array([torque, angle])

w = (np.random.rand(1, k.shape[0])*2 - 1)*w_max
v = (np.random.rand(2, k.shape[0], 1)*2 - 1)
v[:, 0] *= v1_max
v[:, 1] *= v2_max

R = reduce_zonotope(zonotope_inclusion(R), order)
R_theta = reduce_zonotope(zonotope_inclusion(R_theta), order)
R_till = reduce_zonotope(zonotope_inclusion(R_till), order)

R_w = np.array([
    [[-w_max, w_max]]
])
R_w = reduce_zonotope(zonotope_inclusion(R_w), order)

R_v = np.array([
    [[-v1_max, v1_max]],
    [[-v2_max, v2_max]]
])
R_v = reduce_zonotope(zonotope_inclusion(R_v), order) # 2

### H infinity observer design
G_cell = []
for rule in range(len(A_cell)):
    G_cell.append(G)

h_inf_norm, N_L, M_L = h_inf_observer(len(A_cell), A_cell, C, G_cell, E_w, E_vv)
print(f"The H_ifty norm is {h_inf_norm}")

### Simulation
history = [[x_0, c_0, R, R_theta, R_till, np.zeros(shape=(3,2))]]
history_h_ifty = [[x_0, c_0, R, R_theta, R_till, np.zeros(shape=(3,2))]]
# for i in range(nk):
#     iteration = vehicle_simulation(k[i], u[:, i], w[:, i], v[:, i], history[i][0], history[i][1], history[i][2], 
#                                    history[i][3], history[i][4], A_cell, E_d, E_vv, C, G, E_w, R_w, R_v)
#     history.append(iteration)

for i in range(nk):
    iteration = vehicle_simulation(k[i], u[:, i], w[:, i], v[:, i], history_h_ifty[i][0], history_h_ifty[i][1], history_h_ifty[i][2], 
                                   history_h_ifty[i][3], history_h_ifty[i][4], A_cell, E_d, E_vv, C, G, E_w, R_w, R_v, N_L, M_L)
    history_h_ifty.append(iteration)

history = history_h_ifty # TODO: remove - this is just for test

# Plotting
x_history = np.array([item[0] for item in history])
c_history = np.array([item[1] for item in history])
R_history_x1 = np.array([reduce_zonotope(calculate_rs(item[2]), 3)[0][0] for item in history])
R_history_x2 = np.array([reduce_zonotope(calculate_rs(item[2]), 3)[1][1] for item in history])
R_history_x3 = np.array([reduce_zonotope(calculate_rs(item[2]), 3)[2][2] for item in history])
R_theta_history = np.array([item[3] if item[3].shape[1] > 3 else np.zeros(shape=(1,4)) for item in history])
L_history = np.array([item[5] for item in history])

c_top_x1 = np.array([c+interval for c, interval in zip(c_history[:,0], R_history_x1)])
c_low_x1 = np.array([c-interval for c, interval in zip(c_history[:,0], R_history_x1)])

c_top_x2 = np.array([c+interval for c, interval in zip(c_history[:,1], R_history_x2)])
c_low_x2 = np.array([c-interval for c, interval in zip(c_history[:,1], R_history_x2)])

c_top_x3 = np.array([c+interval for c, interval in zip(c_history[:,2], R_history_x3)])
c_low_x3 = np.array([c-interval for c, interval in zip(c_history[:,2], R_history_x3)])

plt.figure(1)
plt.plot(k[:nk], x_history[:nk,0], 'k')
plt.plot(k[:nk], c_history[:nk,0], 'm:')
plt.plot(k[:nk], c_top_x1[:nk], 'b--')
plt.plot(k[:nk], c_low_x1[:nk], 'b--')
# plt.plot(k[:nk], R_theta_history[:nk])
# plt.ylim([4.7, 5.2])
# plt.xlim([0, 1])

plt.figure(2)
plt.plot(k[:nk], x_history[:nk,1], 'k')
plt.plot(k[:nk], c_history[:nk,1], 'm:')
plt.plot(k[:nk], c_top_x2[:nk], 'b--')
plt.plot(k[:nk], c_low_x2[:nk], 'b--')

plt.figure(3)
plt.plot(k[:nk], x_history[:nk,2], 'k')
plt.plot(k[:nk], c_history[:nk,2], 'm:')
plt.plot(k[:nk], c_top_x3[:nk], 'b--')
plt.plot(k[:nk], c_low_x3[:nk], 'b--')

# plt.figure(3)
# plt.plot(k, w[0, :])

# plt.figure(4)
# plt.plot(k, torque)
# plt.figure(5)
# plt.plot(k, angle)

plt.show()