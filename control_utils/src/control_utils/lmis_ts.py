import numpy as np
import math

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