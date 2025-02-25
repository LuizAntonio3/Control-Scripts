import numpy as np

def W_ind(z: np.ndarray[float], z_interval: list[list[float]], mindex: np.ndarray) -> float: #TODO: put this in the same as the variables z
    '''
    :param z: current value for the parameters
    :param z_interval: interval set of the parameters
    :param mindex: Single multi-index

    :return: product of the parameter based functions
    :rtype float:
    '''

    w0_alpha = lambda j, z: ((z_interval[j][1] - z[j]) / (z_interval[j][1] - z_interval[j][0]))
    w1_alpha = lambda j, z: 1 - w0_alpha(j, z)

    w_alpha = lambda j, z: [w0_alpha(j, z), w1_alpha(j, z)]

    prod = 1
    for k in range(len(z)):
        prod *= w_alpha(k, z)[mindex[k]]
    
    return prod

# TODO: put this in a separated module
def L_alpha(z: np.ndarray, z_interval: list[list[float]], mindexes: np.ndarray, L_cell: list[np.ndarray]) -> float:
    ''' Luenberger observer gain for a politopic representation
    :param z: current value for the parameters
    :param z_interval: interval set of the parameters
    :param mindexes: Set of multi-indexes
    :param L_cell: Gain L for each vertex

    :return: observer gain
    :rtype float:
    '''
    
    L = 0

    for i in range(len(mindexes)):
        L += W_ind(z, z_interval, mindexes[i])*L_cell[i]

    return L

def permn(V: np.ndarray, N: int, K = None) -> tuple[np.ndarray, np.ndarray]:
    '''
    :param V: Indexes of the sums
    :type V: np.ndarray
    :para N: number of sums of the given indexes

    :return: [Combination, Indexes]
    :rtype: tuple[np.ndarray, np.ndarray]
    '''
    if N < 0:
        raise("Second argument should be a positive interger")
    
    nV = V.shape[0]
    M = np.zeros(shape=(nV, N))
    I = np.zeros(shape=(nV, N))
    if K == None:
        # Return all permutations

        if nV == 0 or N == 0:
            M = np.zeros(shape=(nV, N))
            I = np.zeros(shape=(nV, N))
        elif N == 1:
            M = V.reshape((nV, 1))
            I = np.arange(nV).T
        else:
            I = np.flip(np.array(np.meshgrid(*[np.arange(nV) for i in range(N)], indexing="ij")).T.reshape(-1, N), axis=1)
            M = V[I]
    else:
        # not implemented
        pass
    return [M, I]

# old bin_from_perm
def bin_perm_to_dec(perm: np.ndarray) -> int:
    ''' Given a binary multi-index (the case for interval sets) retrieves its representation as an decimal
    :param perm: a single multi-index
    :type perm: np.ndarray

    :return: decimal representation of a given binary multindex
    :rtype: int
    '''
    return int(str(perm).strip("[]").replace(" ", ""), 2)

def perm_plus(t_perm: np.ndarray) -> np.ndarray:
    ''' 
        :param t_perm: Set of multi-indexes
        :type t_perm: np.ndarray

        :return: Triangular multi-indexes of a given set of multi-indexes
        :rtype: np.ndarray
    '''
    t_perm_plus = []
    for i in range(len(t_perm)):
        add = True
        for j in range(len(t_perm[i]) - 1):
            if not t_perm[i][j] <= t_perm[i][j+1]:
                add = False
                break

        if add:
            t_perm_plus.append(t_perm[i])
    
    return np.array(t_perm_plus)

def multi_index_permutation(mindex: np.ndarray[float]) -> np.ndarray:
    '''
    this function takes a multi-index and retrieves its permutation -> e.g.: 0001 generates 0001, 0010, 0100, 1000

    :param mindex: a single multi-iindex
    
    :return: set of permutation of the given multi-index
    :rtype: np.ndarray
    '''

    return np.unique(list(heap_permutations(mindex, len(mindex))), axis=0)

def heap_permutations(a, n):
    """
    Generate all permutations of list `a` using Heap's algorithm.
    
    Parameters:
        a (list): The list to permute.
        n (int): The length of the list to consider (usually len(a)).
        
    Yields:
        list: A permutation of the list `a`.
    """
    if n == 1:
        # When n is 1, yield a copy of the current permutation.
        yield a.copy()
    else:
        for i in range(n):
            # Recursively generate permutations for n-1 elements.
            yield from heap_permutations(a, n - 1)
            # Depending on whether n is even or odd, swap accordingly.
            if n % 2 == 0:
                a[i], a[n - 1] = a[n - 1], a[i]
            else:
                a[0], a[n - 1] = a[n - 1], a[0]