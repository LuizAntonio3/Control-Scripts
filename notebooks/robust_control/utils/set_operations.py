import numpy as np

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

    return np.concat([mid_R, S], axis=1)

def reduce_zonotope(R: np.array, q):
    indexlist = np.argsort(np.linalg.norm(R, axis=0).T)
    R_sorted: np.array = R[:, np.flip(indexlist)]
    
    n = R_sorted.shape[0]
    p = R_sorted.shape[1]

    if p <= q:
        return R
    
    r_index = q-n
    if r_index < 0:
        r_index = 0
    
    R_right = R_sorted[:, 0:r_index]
    R_left = R_sorted[:, q-n:p] # in the paper there was a "q-n+1" but it was index 1 related - changed to q-n
    # print(R_left)

    b_R = np.diag(np.abs(R_left@np.ones(shape=(R_left.shape[1], 1))).squeeze())
    # print(R_left@np.ones(shape=(R_left.shape[1], 1)))

    # print(R_right)
    # print(b_R)

    return np.concat([R_right, b_R], axis=1)