import time
import numpy as np
import scipy.sparse as sparse

# The purpose of this file is to identify which way of computing the gradient descent
# iteration for the softmax method is faster. More specifically, we will compare
# three different methods to get the loss derivative matrix

ITER = 100
K = 10
N = 10000

def for_loop_approach(Y: np.ndarray, k: int) -> np.ndarray:
    """
    :param Y: (n, ), labels for each feature vector in the training set
    :param k: int, indicating how many possible labels there are
    :return: M: (k, n) such that M[i, j] == 1 if Y[i] == j else 0
    """
    n = Y.shape[0]
    M = np.zeros((k, n))
    for i in range(k):
        for j in range(n):
            if Y[j] == i:
                M[i, j] = 1

    return M.T


def opt_for_approach(Y: np.ndarray, k: int) -> np.ndarray:
    """
    See docstring for for_loop_approach
    """
    n = Y.shape[0]
    M = np.ones((k, n))

    for j in range(k):
        M[j, :] = (Y==j)

    return M


def comprehension_approach(Y: np.ndarray, k: int) -> np.ndarray:
    """
    See docstring for for_loop_approach
    """
    M = np.array([[1 if i == j else 0 for j in range(k)] for i in Y])
    return M.T


def np_adv_index_approach(Y: np.ndarray, k: int) -> np.ndarray:
    """
    See docstring for for_loop_approach
    """
    n = Y.shape[0]
    Y_idx = np.arange(n)  # list of indeces in Y
    M = np.zeros((k, n))
    M[Y, Y_idx] = 1
    return M


def coo_matrix_approach(Y: np.ndarray, k: int) -> np.ndarray:
    """
    See docstring for for_loop_approach
    """
    n = Y.shape[0]
    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n)).toarray()
    return M


def main():
    Y = np.random.randint(0, K, size=N)
    func_dict = {'for loop': for_loop_approach,
                 'optimized for loop': opt_for_approach,
                 'comprehension': comprehension_approach,
                 'advanced indexing': np_adv_index_approach,
                 'coo sparse matrix': coo_matrix_approach}

    M_list = []
    for name, func in func_dict.items():
        t0 = time.time()
        for i in range(ITER):
            M = func(Y, K)
        tf = time.time()
        print(f"Time for {name}: {tf - t0}")
        M_list.append(M)

    for M in M_list:
        print(M, f"{M.shape=}")


if __name__ == '__main__':
    main()
