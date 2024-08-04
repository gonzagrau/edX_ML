import numpy as np
from scipy.stats import norm


def main():
    N_CLUSTERS = 2
    N_POINTS = 5
    mu = np.array([-3, 2])
    var = np.array([4, 4])
    sd = np.sqrt(var)
    p = np.array([0.5, 0.5])
    points = np.array([0.2, -0.9, -1, 1.2, 1.8])
    p_j = np.zeros((N_CLUSTERS, N_POINTS))

    for i, x in enumerate(points):
        # Get posteriors
        numerators = [p[j]*norm(mu[j], sd[j]).pdf(x)
                      for j in range(N_CLUSTERS)]
        numerators = np.array(numerators)
        denom = numerators.sum()
        posterior = numerators/denom
        print(f"x_{i+1}={x}")
        print(f"p(1 | {i+1}) = {posterior}")
        p_j[:, i] = posterior

    # Update params
    p_upd = p_j.sum(axis=1)/N_POINTS
    mu_upd = np.sum(p_j*points, axis=1)/p_j.sum(axis=1)
    std_upd = np.sum(p_j[0, :] * (points - mu_upd[0])**2) / p_j[0, :].sum()
    print(f"{p_upd=}")
    print(f"{mu_upd=}")
    print(f"{std_upd=}")

if __name__ == '__main__':
    main()