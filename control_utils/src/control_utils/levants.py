import numpy as np

def sign(value) -> int:
    """Returns the sign of a given function"""

    # return 2/(1+math.exp(-value)) - 0.5

    if value < 0:
        return -1
    elif value == 0:
        return 0
    else:
        return 1

def differentiator(y: float, z: np.ndarray, lamb: np.ndarray) -> np.ndarray:
    """Returns the levants differentiator of a given signal
    
    Parameters
    ----------
    y : float
        Signal
    z : np.ndarray
        Levants differentiator integration
    lamb : np.ndarray
        Gains for the differentiator
    """
    n = lamb.shape[0] - 1
    v = np.zeros(shape=(n, 1))
    dz = np.zeros(shape=(n+1, 1))

    for i in range(n):
        exp = (n-i)/(n+1-i)

        if i == 0:
            v[i]= -lamb[i]*((abs(z[i] - y))**exp)*sign(z[i] - y) + z[i+1]
        else:
            v[i]= -lamb[i]*((abs(z[i] - v[i-1]))**exp)*sign(z[i] - v[i-1]) + z[i+1]

    dz[0:n] = v
    dz[n] = -lamb[n]*sign(z[n] - v[n-1])

    return dz

if __name__ == "__main__":
    lamb = np.array([4, 3, 2, 1])
    z = np.array([0, 0, 0, 0])
    y = 2.0

    dz = differentiator(y, z, lamb)

    print(dz)
    