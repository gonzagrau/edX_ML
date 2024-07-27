import numpy as np
from typing import Tuple, List, Callable
from problem1 import mistake_perceptron


@np.vectorize(signature='(n)->(k)')
def quad_feature(x: np.ndarray) -> np.ndarray:
    """
    Quadratic kernel
    :param x: 2 element array
    :return: 3 element array
    """
    return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])


@np.vectorize(signature='(m),(m)->()')
def quad_kernel(x_1: np.ndarray, x_2: np.ndarray) -> float:
    """
    Given two input feature vector, returns the dot product of their higher dimensional embedding
    after applying the quadratic featurization
    :param x_1: first feature vector, N-element array
    :param x_2: second feature vector, N-element array
    :return: flaot indicating phi(x_1).T @ phi(x_2)
    """
    return (x_1.T @ x_2)**2


def kernel_predict(points: np.ndarray,
                   labels: np.ndarray,
                   alpha: np.ndarray,
                   kernel_func: Callable) -> np.ndarray:
    """
    Uses the alpha vector from the kernel perceptron algorithm to predict from kernels
    :param points: feature vectors in a matrix along axis=0
    :param labels: training set labels
    :param alpha: indicates weight for each dot product
    :param kernel_func: kernel_func(x, x') = phi(x).T @ phi(x')
    :return:
    """
    N, M = points.shape
    pred = np.zeros(N)
    for i in range(N):
        pred += kernel_func(points, points[i, :])*labels
    return np.sign(pred)


def predict(points, theta, theta_0) -> np.ndarray:
    """
    Perform linear classification prediction
    :param points: feature matrix, nxm 2D matrix
    :param theta: normal vector to the hyperplane, array with m elements
    :param theta_0: offset, a flaot
    :return: predictions for all points
    """
    leeway = points @ theta + theta_0
    return np.sign(leeway)


def ex_1(points, labels, mistakes):
    ### Runs perceptron on higher dimensional embedding of features
    # trans_points = quad_feature(points)
    # ex_point = trans_points[-1, :]
    # for i in range(points.shape[0]):
    #     wo_kernel = trans_points[i, :].T @ ex_point
    #     with_kernel = quad_kernel(points[i, :], points[-1, :])
    #     print(points[i, :], points[-1, :])
    #     print(trans_points[i, :], ex_point)
    #     print(f"{wo_kernel=}")
    #     print(f"{with_kernel=}")
    #     print()
    # theta, theta_0 = mistake_perceptron(trans_points, labels, mistakes, prints=False)
    # print(f"{theta=}")
    # print(f"{theta_0=}")
    # pred = predict(trans_points, theta, theta_0)
    # print(pred)
    # accuracy = pred == labels
    # print(accuracy)
    #
    ### Now using kernel perceptron
    pred_2 = kernel_predict(points, labels, alpha=mistakes, kernel_func=quad_kernel)
    print(pred_2)
    acc_2 = pred_2 == labels
    print(acc_2)

def main():
    labels = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    points = np.array([[0,0],
                       [2,0],
                       [1,1],
                       [0,2],
                       [3,3],
                       [4,1],
                       [5,2],
                       [1,4],
                       [4,4],
                       [5,5]])
    mistakes = np.array([1, 65, 11, 31, 72, 30, 0, 21, 4, 15])
    ex_1(points, labels, mistakes)


if __name__ == '__main__':
    main()
