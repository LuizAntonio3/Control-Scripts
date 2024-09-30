import control as ct
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

### Autonomous ground vehicle

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
v_y = [-1.5, 1.5]   # [m/s]
r_interval = [-0.55, 0.55] # [rad/s]

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
# f_v_cell = []

for V_x_i in V_x_interval:
    for r_i in r_interval:
        A_cell.append(A_xi(V_x_i, r_i))
        # f_v_cell.append(f_v_xi(V_x_i, r_i))

E_v = np.array([
    [1/I_e, 0],
    [0, 2*C_f/M],
    [0, 2*l_f*C_f/I_z]
])

C = np.array([
    [1, 0, 0],
    [0, 0, 1]
])

G_v = np.array([0, -C_y/M, 0]).reshape(3, 1)

E_w = np.array([1, 0, 0]).reshape(3, 1)

#### Discrete Time Matrices
for i in range(len(A_cell)):
    A_cell[i] = Ts * A_cell[i] + np.eye(A_cell[i].shape[0])
    # f_v_cell[i] = Ts * f_v_cell[i]
f_v_xi_d = lambda V_x, r: Ts*f_v_xi(V_x, r)
E_d = Ts * E_v
G = Ts * G_v
E_w = Ts * E_w

omega_v1 = lambda V_x: (V_x_interval[1] - V_x)/(V_x_interval[1] - V_x_interval[0])
omega_v2 = lambda V_x: (V_x - V_x_interval[0])/(V_x_interval[1] - V_x_interval[0])
omega_r1 = lambda r: (r_interval[1] - r)/(r_interval[1] - r_interval[0])
omega_r2 = lambda r: (r - r_interval[0])/(r_interval[1] - r_interval[0])

h_xi = lambda V_x, r: [
    omega_v1(V_x)*omega_r1(r), 
    omega_v1(V_x)*omega_r2(r), 
    omega_v2(V_x)*omega_r1(r), 
    omega_v2(V_x)*omega_r2(r)
]

# ---------------------------------------------------------------------------

def calculate_rs(R):
    n = R.shape[0]
    intervals = R.shape[1]

    rs = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(intervals):
            rs[i, i] += np.abs(R[i, j])

    return rs

def fastNN(A, B):
    m = A.shape[0]
    n = B.shape[1]

    u = np.zeros(shape=(1, A.shape[1]))
    v = np.zeros(shape=(B.shape[0], 1))

    for i in range(u.shape[1]):
        u[0, i] = np.max(A[:,i])

    for i in range(v.shape[0]):
        v[i, 0] = np.max(B[i, :])

    S = np.ones(shape=(m, 1))@(u@B)
    T = (A@v)@np.ones(shape=(1, n))

    W = np.zeros(shape=(m, n))

    for i in range(m):
        for j in range(n):
            W[i, j] = np.min([S[i, j], T[i, j]])
    
    return W

def build_interval_from_bounds(lower, upper):
    interval = np.zeros(shape=(lower.shape[0], lower.shape[1], 2))
    
    for i in range(lower.shape[0]):
        for j in range(lower.shape[1]):
            interval[i, j] = [lower[i, j], upper[i, j]]

    return interval

    # return np.array([[
    #     [lower[i, j], upper[i, j]] 
    #         if lower[i, j] < upper[i, j] 
    #         else [upper[i, j], lower[i, j]] 
    #         for i in range(lower.shape[0])] 
    #         for j in range(lower.shape[1])
    # ])

def m_interval(interval):
    return (interval[1] + interval[0])/2

def w_interval(interval):
    return (interval[1] - interval[0])/2

def interval_product(A, B):

    mid_A = np.zeros(shape=(A.shape[0], A.shape[1]))
    wid_A = np.zeros(shape=(A.shape[0], A.shape[1]))
    mid_B = np.zeros(shape=(B.shape[0], B.shape[1]))
    wid_B = np.zeros(shape=(B.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            mid_A[i, j] = m_interval(A[i, j])
            wid_A[i, j] = w_interval(A[i, j])

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            mid_B[i, j] = m_interval(B[i, j])
            wid_B[i, j] = w_interval(B[i, j])

    mid_C = mid_A@mid_B
    # wid_C = np.max([np.min(wid_A[np.nonzero(wid_A)]) , np.min(wid_B[np.nonzero(wid_B)])]) # check this calculation
    wid_C = fastNN(wid_A, np.abs(mid_B)+wid_B) + fastNN(np.abs(mid_A), wid_B)

    C_low = mid_C - wid_C
    C_high = mid_C + wid_C

    return build_interval_from_bounds(C_low, C_high)

def zonotope_inclusion(generators):
    n = generators.shape[0]
    intervals = generators.shape[1]

    mid_R = np.zeros(shape=(n, intervals))
    S = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(intervals):
            mid_R[i, j] = m_interval(generators[i, j])
            S[i, i] += w_interval(generators[i, j])

    # print(f"mid: {mid_R}")
    # print(f"\nS: {S}")

    return np.concat([mid_R, S], axis=1)

def reduce_zonotope(R: np.array, q):
    indexlist = np.argsort(np.linalg.norm(R, axis=0).T)
    R_sorted: np.array = R[:, np.flip(indexlist)]
    
    n = R_sorted.shape[0]
    p = R_sorted.shape[1]

    if p <= q:
        return R
    
    R_right = R_sorted[:, 0:q-n]
    R_left = R_sorted[:, q-n:p] # in the paper there was a "q-n+1" but it was index 1 related - changed to q-n
    # print(R_left)

    b_R = np.diag(np.abs(R_left@np.ones(shape=(R_left.shape[1], 1))).squeeze())
    # print(R_left@np.ones(shape=(R_left.shape[1], 1)))

    # print(R_right)
    # print(b_R)

    return np.concat([R_right, b_R], axis=1)

def retrieve_R_theta(A_hat, G_h, E_w, R_till, R_theta, R_w):
    # compute R_till
    # R in this is R_till <---------------
    R_till_plus = np.concat([A_hat@R_till, G_h@R_theta, E_w@R_w, np.zeros(shape=(3, 2))], axis=1) # use k=0 to calculate k=1

    # compute the boundaries of x_till
    x_high = +calculate_rs(R_till_plus) # check the number of intervals
    x_low = -calculate_rs(R_till_plus)  # check the number of intervals
    
    R_till_plus = reduce_zonotope(R_till_plus, 3)

    # x_till_interval = np.array([[
    #     [x_low[i, j], x_high[i, j]] 
    #         if x_low[i, j] < x_high[i, j] 
    #         else [x_high[i, j], x_low[i, j]] 
    #         for j in range(x_low.shape[0])] 
    #         for i in range(x_high.shape[0])
    # ])

    x_till_interval = build_interval_from_bounds(x_low, x_high)

    phi_interval = np.array([
        [[0, 0], [-3, 3], [0, 0]]
    ])

    # erro nesse calculo aqui devido a dimensões diferentes de phi_interval e x_till_interval
    R_theta_interval = interval_product(phi_interval, x_till_interval)
    R_k_theta = zonotope_inclusion(R_theta_interval)

    return [R_till_plus, R_k_theta]

def vehicle_simulation(k, u, w, x, c, R, R_theta, R_till, A_cell, E_d, C, G, E_w, R_w, L_h = None):
    # x  = [v_x, v_y, r]
    # xi = [1/v_x, r]
    # c_w and R_w unknown -> give any bounded value to R_w -> an interval set may be enough?

    # membership update
    h = h_xi(1/x[0], x[2])
    f_xi = f_v_xi_d(1/x[0], x[2])
    A_h = 0
    G_h = 0

    phi_x = lambda par: par[1]**2
    
    # Model Simulation
    x_plus = 0
    for i in range(len(h)):
        # x_plus += h[i]*A_cell[i]@x + E_d@u + f_xi + G*phi_x(x) + E_w*w
        x_plus += (h[i]*A_cell[i]@x).reshape(3,1)
        A_h += h[i] * A_cell[i]
        G_h += h[i] * G
    
    x_plus = x_plus.reshape(3, 1)
    x_plus += E_d@u + f_xi + G*phi_x(x) + E_w*w
    y = C@x # x_plus ???
    # --------------------------------------

    # Estimator gain design
    # if len(R.shape) > 2:
    #     R = R.reshape((R.shape[0], R.shape[1]*R.shape[2]))

    if L_h is None:
        P_till = R@R.T
        omega = C@P_till@C.T + 0
        Psi_h = A_h@P_till@C.T
        L_h = Psi_h@np.linalg.inv(omega)
    # else use gain provided by H-infty filter
    
    # R_plus = reduce_zonotope(R, 3)

    # State estimation
    A_hat = A_h - L_h@C
    [R_till_plus, R_theta] = retrieve_R_theta(A_hat, G_h, E_w, R_till, R_theta, R_w)

    f_xi = f_v_xi_d(1/c[0], c[2])
    c_plus = A_hat@c + E_d@u + f_xi + G_h*phi_x(c) + L_h@y
    R_plus = np.concat([A_hat@R, G_h@R_theta, E_w@R_w, np.zeros(shape=(3, 1))], axis=1) # problably gonna have to fix this 0 to be a matrix

    R_plus = reduce_zonotope(R_plus, 3)

    # R reduction

    return [x_plus, c_plus, R_plus, R_theta, R_till_plus]

### Simulation
k = np.arange(0, Ts*30, Ts)
torque = np.concat([
    5*np.ones(shape=(k.shape[0]//2, 1)),
    10*np.ones(shape=(k.shape[0]//2 + k.shape[0]%2, 1))
])

angle = np.concat([
    np.zeros(shape=(k.shape[0]//5, 1)),
    -.75*np.ones(shape=(k.shape[0]//5, 1)),
    np.zeros(shape=(k.shape[0]//5, 1)),
    1*np.ones(shape=(k.shape[0]//5 + k.shape[0]%5, 1)),
    np.zeros(shape=(k.shape[0]//5, 1))
])

u = np.array([torque, angle])

np.random.seed(2109)
w_max = .15
w = (np.random.rand(1, k.shape[0])*2 - 1)*w_max

last_mod = 0
for i in range(0, w.shape[1]):
    if i % 16 == 0:
        last_mod = i

    w[0, i] = w[0,last_mod]

x_0 = np.array([5, -1, 0]).reshape(3, 1)
c_0 = x_0
R = np.array([
    [[-.2, .2]],
    [[-.5, .5]],
    [[-.5, .5]]
])
R = reduce_zonotope(zonotope_inclusion(R), 3)

# check R_theta and phi_interval and product
R_theta = np.array([[-1e6, 1e6]]).reshape(1, 1, 2)
R_theta = reduce_zonotope(zonotope_inclusion(R_theta), 3)

R_till = R # maybe change this

R_w = np.array([
    [[-w_max, w_max]]
])
R_w = reduce_zonotope(zonotope_inclusion(R_w), 3)

history = [[x_0, c_0, R, R_theta, R_till]]
# print(history[0][2])

nk = k.shape[0]-1
nk = 30

for i in range(nk):
    iteration = vehicle_simulation(k[i], u[:, i], w[:, i], history[i][0], history[i][1], history[i][2], 
                                   history[i][3], history[i][4], A_cell, E_d, C, G, E_w, R_w)
    history.append(iteration)

# Plotting

x_history = np.array([item[0] for item in history])
c_history = np.array([item[1] for item in history])
R_history_x1 = np.array([item[2][0][0] for item in history])
R_history_x2 = np.array([item[2][1][1] for item in history])
R_theta_history = np.array([item[3] if item[3].shape[1] > 3 else np.zeros(shape=(1,4)) for item in history])

c_top_x1 = np.array([c+interval for c, interval in zip(c_history[:,0], R_history_x1)])
c_low_x1 = np.array([c-interval for c, interval in zip(c_history[:,0], R_history_x1)])

c_top_x2 = np.array([c+interval for c, interval in zip(c_history[:,1], R_history_x2)])
c_low_x2 = np.array([c-interval for c, interval in zip(c_history[:,1], R_history_x2)])

plt.figure(1)
plt.plot(k[:nk], x_history[:nk,0], 'k')
# plt.plot(k[:nk], c_history[:nk,0], 'b')
plt.plot(k[:nk], c_top_x1[:nk], 'b--')
plt.plot(k[:nk], c_low_x1[:nk], 'b--')
# plt.plot(k[:nk], R_theta_history[:nk])
# plt.ylim([4.7, 5.2])
# plt.xlim([0, 1])

plt.figure(2)
plt.plot(k[:nk], x_history[:nk,1], 'k')
# plt.plot(k[:nk], c_history[:nk,1], 'b')
plt.plot(k[:nk], c_top_x2[:nk], 'b--')
plt.plot(k[:nk], c_low_x2[:nk], 'b--')

# plt.figure(3)
# plt.plot(k, w[0, :])

# plt.figure(4)
# plt.plot(k, torque)
# plt.figure(5)
# plt.plot(k, angle)

plt.show()