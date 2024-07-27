import numpy as np


def mistake_perceptron(points, labels, mistakes, prints=True):
    # Perform linear perceptron algorithm
    theta = np.zeros(points.shape[1])
    theta_0 = 0
    for i in range(points.shape[0]):
        label, mistake, point = labels[i], mistakes[i], points[i, :]
        theta += mistake * (label * point)
        theta_0 += mistake * label
        if prints:
            print(f"{label=}, {point=}, {mistake=}")
            print(f"{theta=}, {theta_0=}", '\n\n')
    return theta, theta_0



def hinge_loss(point: np.ndarray,
               label: float,
               theta: np.ndarray,
               theta_0: float) -> float:
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.
    """
    return np.max([0, 1 - label * (theta.T @ point + theta_0)])


def ex_2(labels, points, mistakes, theta, theta_0):
    # Find total hinge loss
    loss = 0
    for i in range(points.shape[0]):
        label, mistake, point = labels[i], mistakes[i], points[i, :]
        pred = theta.T @ point + theta_0
        new_loss = np.max([0, 1 - label*pred])
        print(f"pred={np.sign(pred)}, {label=}")
        print(f"loss at {point}: {new_loss}")
        loss += new_loss
    print(loss)
    return loss


def main():
    labels = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
    points = [[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]]
    mistakes = [1, 9, 10, 5, 9, 11, 0, 3, 1, 1]
    points = np.array(points)
    mistake_perceptron(points, labels, mistakes)

    theta = np.array([1, 1])/2
    theta_0 = -5/2
    ex_2(labels, points, mistakes, theta, theta_0)


if __name__ == '__main__':
    main()