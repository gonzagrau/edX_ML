import numpy as np
from scipy.stats import norm

def main():
    x = np.array([-1, 0, 4, 5, 6])
    n = x.shape[0]
    k = 2
    pi = np.array([0.5, 0.5])
    mu = np.array([6, 7])
    sig = np.array([1, 2])
    theta = np.concatenate([pi, mu, sig**2])

    # p_y = np.zeros((n, k))
    gauss_k = np.array([norm(mu[i], sig[i]).pdf(x) for i in range(k)], dtype=np.float64)
    print(gauss_k)
    p_y = np.dot(pi, gauss_k)
    log_l = np.sum(np.log(p_y))
    print(f"{log_l=}")

    clus_weight = gauss_k * pi.reshape((2, 1))
    p_y_k = clus_weight / clus_weight.sum(axis=0)
    print(f"{p_y_k=}")
    print((p_y_k[0, :] > p_y_k[1, :]))


if __name__ == '__main__':
    main()