import numpy as np

def ReLU(z):
    return np.max(z, 0)

def ReLU_prime(z):
    return int(z > 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return np.exp(-z) / (1 + np.exp(-z))**2


def cost(y, t):
    return 0.5*(y-t)**2

def cost_prime(y, t):
    return (y-t)

def main():
    # params
    w1 = 0.01
    w2 = -5
    b = -1
    x = 3
    t = 1
    # layer 1
    z1 = w1*x
    a1 = ReLU(z1)
    # layer 2
    z2 = w2*a1 + b
    y = sigmoid(z2)
    # cost
    C = cost(y, t)
    print(C)
    # derivative w1
    delta_2 = cost_prime(y, t)*sigmoid_prime(z2)
    dc_dw1 = delta_2*w2*ReLU_prime(z1)*x
    dc_dw2 = delta_2*a1
    dc_db = delta_2*1

    print(f"{dc_dw1=}")
    print(f"{dc_dw2=}")
    print(f"{dc_db=}")

if __name__ == '__main__':
    main()