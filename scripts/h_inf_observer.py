from utils.lmis_ts import *
import cvxpy as cp

def h_inf_observer(n_rules: int, A: np.ndarray, C: np.ndarray, G: np.ndarray, E_w: np.ndarray, E_v: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # TODO: consider G as a list

    sums = 3
    nx = A[0].shape[0]
    ny = C.shape[0]

    P: list[cp.Variable] = []
    N: list[cp.Variable] = []
    M: list[cp.Variable] = []

    for _ in range(n_rules):
        P.append(cp.Variable((nx, nx), symmetric=True))
        N.append(cp.Variable((nx, nx), symmetric=True))
        M.append(cp.Variable((nx, ny)))
    
    gamma = cp.Variable(nonneg=True)

    _, indexes = permn(np.arange(n_rules), sums)

    Gamma = None # fix this part
    for ind in indexes:
        i, j, q = ind # TODO: check if this is correct

        # intermediate values
        Gamma_51: np.ndarray = N[i]@A[i] - M[i]@C
        Gamma_52: np.ndarray = N[i]@G[j]
        Gamma_53: np.ndarray = N[i]@E_w
        Gamma_54: np.ndarray = -M[i]@E_v
        Gamma_55: np.ndarray = N[i] + N[i].T - P[q]

        I = np.eye(nx, nx)
        

        Gamma_i = cp.bmat([
            [P[i] - I        , np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 2)), Gamma_51.T],
            [np.zeros((1, 3)), gamma*np.eye(1) , np.zeros((1, 1)), np.zeros((1, 2)), Gamma_52.T],
            [np.zeros((1, 3)), np.zeros((1, 1)), gamma*np.eye(1) , np.zeros((1, 2)), Gamma_53.T],
            [np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((2, 1)), gamma*np.eye(2) , Gamma_54.T],
            [Gamma_51        , Gamma_52        , Gamma_53        , Gamma_54        , Gamma_55]
        ])

        if Gamma is None:
            shape_aux = Gamma_i.shape
            Gamma = np.empty(shape=(*[n_rules for i in range(sums)], *shape_aux), dtype=object)
        
        Gamma[i, j, q, :, :] = Gamma_i

    eps = 1e-9
    lmis = []

    for p in P:
        lmis += [p >> eps]

    lmi_indexes = permutations_for_lmis(indexes, n_rules)

    for lmi_ind in lmi_indexes:
        gamma_aux = 0
        n_sums = lmi_ind.shape[0]
        for ns in range(n_sums): 
            i, j, q = lmi_ind[ns]

            if i == j and n_sums > 1:
                gamma_aux += 1/(n_rules - 1) * Gamma[i, j, q][0, 0]
            else:
                gamma_aux += Gamma[i, j, q][0, 0]
            
        lmis += [gamma_aux >> eps]

    problem = cp.Problem(cp.Minimize(gamma), constraints=lmis)
    result = problem.solve(solver=cp.MOSEK, verbose=True)

    return (gamma.value)**(1/2), [n.value for n in N], [m.value for m in M]