import numpy as np

def perceptron_algorithm(X: np.ndarray,
                         Y: np.ndarray,
                         N_iter: int=10,
                         theta_init: np.ndarray | None=None,
                         theta_0_init: float=0) -> np.ndarray:
    """
    Applies the perceptron algorithm to linearly separate
    the training data
    :param X: 2D array where each column is a feature vector
    :param Y: labels
    :param N_iter: maximum number of iterations
    :param theta_init: initial normal vector
    :param theta_0_init: initial offset
    :return: theta, the normal vector to the separating hyperplane
    """
    M, N = X.shape
    if theta_init is None:
        theta = np.zeros(M)
    else:
        theta = theta_init
    theta_0 = theta_0_init
    error_count = 0
    for k in range(N_iter):
        n_mis = 0
        for i in range(N):
            x = X[:, i]
            y = Y[i]
            if (theta.T @ x + theta_0) * y <= 0:
                error_count += 1
                n_mis +=1
                print('error', error_count)
                print(f"{x=}, {y=}")
                theta += x*y
                print(f"{theta=}")
                theta_0 += y
                print(f"{theta_0=}", '\n')
        if n_mis == 0:
            print(f"convergence finished after {k} iterations")
            break
    return theta


def perceptron_no_offset(X: np.ndarray,
                         Y: np.ndarray,
                         N_iter: int=10,
                         theta_init: np.ndarray | None=None) -> np.ndarray:
    """
    Applies the perceptron algorithm without offset
    to linearly separate the training data
    :param X: 2D array where each column is a feature vector
    :param Y: labels
    :param N_iter: maximum number of iterations
    :param theta_init: initial normal vector
    :return: theta, the normal vector to the separating hyperplane
    """
    M, N = X.shape
    if theta_init is None:
        theta = np.zeros(M)
    else:
        theta = theta_init
    error_count = 0
    for k in range(N_iter):
        n_mis = 0
        for i in range(N):
            x = X[:, i]
            y = Y[i]
            if (theta.T @ x) * y <= 0:
                error_count += 1
                n_mis +=1
                print('error', error_count)
                print(f"{x=}, {y=}")
                theta += x*y
                print(f"{theta=}")
        if n_mis == 0:
            print(f"convergence finished after {k} iterations")
            break
    return theta


def main():
    X = np.array([[-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]])
    Y = np.array([1, 1, 1])
    theta = perceptron_no_offset(X, Y)
    print(theta)


if __name__ == '__main__':
    main()