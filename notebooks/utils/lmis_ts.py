import numpy as np

def sign(value):
    '''
    Returns the sign of a given function
    '''
    if value < 0:
        return -1
    elif value == 0:
        return 0
    else:
        return 1

def permn(V: np.ndarray, N: int, K = None) -> tuple[np.ndarray, np.ndarray]:
    '''
    V: Indexes of the sums \\
    N: Number of sums
    
    [M, I] -> [Combination, Indexes]
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

def bin_from_perm(perm: np.ndarray):
    return int(str(perm).strip("[]").replace(" ", ""), 2)

def perm_plus(t_perm): # maybe this function should not be in this file
    '''
        this generates the B^q+
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
    
    return t_perm_plus

def multi_index_permutation(index: np.ndarray[float]):
    '''
    this function takes a multiindex -> e.g.: 0001 \n
    and retrieves its permutation
    '''

    return np.unique(list(heap_permutations(index, len(index))), axis=0)

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

def permutations_for_lmis(perm: np.ndarray, r: int, K = None) -> tuple[np.ndarray, np.ndarray]:
    '''
    perm: permutation of indexes \\
    r: number of rules
    '''
    indexes = []
    for comb in perm:
        indexes.append(np.array([comb]))
    	
    for i in range(r):
        for j in range(r):
            if i == j:
                continue
            for q in range(r):
                ind = np.array([
                    [i, i, q],
                    [i, j, q],
                    [j, i, q]
                ])

                indexes.append(ind)

    return indexes