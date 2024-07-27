import numpy as np

@np.vectorize(signature='(n)->(k)')
def quad_feature(x: np.ndarray) -> np.ndarray:
    """
    Quadratic kernel
    :param x: 2-element array
    :return: 3-element array
    """
    return np.array([x[0], np.sqrt(2) * x[0] * x[1], x[1]**2])

# Example 2D array where each row is a 2-element array
array_2d = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])
print(array_2d.shape)
# Apply the vectorized function to the 2D array
result = quad_feature(array_2d)
print(result.shape)
print(result)