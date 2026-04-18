import numpy as np

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
    
if __name__ == "__main__":
    M, I = permn(np.arange(0,3), 3)
    indexes = permutations_for_lmis(M, 3)

    for ind in indexes:
        print(ind)
    print("****")
    print(I)
    print("****")
    print(M)