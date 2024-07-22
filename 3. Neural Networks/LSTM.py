import numpy as np

def sigmoid(z):
    if z >= 1:
        return 1
    if z <= -1:
        return 0
    return 1 / (1 + np.exp(-z))


def tanh(z):
    if z >= 1:
        return 1
    if z <= -1:
        return -1
    return np.tanh(z)


h_init =  0
c_init = 0
x = np.array([1, 1, 0, 1, 1])
# x = np.array([0, 0, 1, 1, 1, 0])
W_fh = 0; W_fx = 0; b_f = -100; W_ch = -100
W_ih = 0; W_ix = 100; b_i = 100; W_cx = 50
W_oh = 0; W_ox = 100; b_0 = 0; b_c = 0

f = np.zeros_like(x)
i = np.zeros_like(x)
o = np.zeros_like(x)
c = np.zeros_like(x)
h = np.zeros_like(x)

f[0] = sigmoid(W_fh*h_init + W_fx*x[0] + b_f)
i[0] = sigmoid(W_ih*h_init + W_ix*x[0] + b_i)
o[0] = sigmoid(W_oh*h_init + W_ox*x[0] + b_0)
c[0] = f[0]*c_init + i[0]*tanh(W_ch*h_init + W_cx*x[0] + b_c)
h[0] = o[0]*tanh(c[0])
if np.abs(h[0]) <= 0.5:
    h[0] = 0

for t in range(1, x.shape[0]):
    f[t] = sigmoid(W_fh*h[t-1] + W_fx*x[t] + b_f)
    i[t] = sigmoid(W_ih*h[t-1] + W_ix*x[t] + b_i)
    o[t] = sigmoid(W_oh*h[t-1] + W_ox*x[t] + b_0)
    c[t] = f[t]*c[t-1] + i[t]*tanh(W_ch*h[t-1] + W_cx*x[t] + b_c)
    h[t] = o[t]*tanh(c[t])
    if np.abs(h[t]) <= 0.5:
        h[t] = 0
#
#     x_curr = x[:t]
#     tot_zeros = np.sum(x_curr == 0)
#     tot_ones = np.sum(x_curr == 1)
#     print(f"{x_curr=}")
#     print(f"{tot_zeros%2=}")
#     print(f"{tot_ones%2=}")
#     print(f"{h[t]=}")
#     print()
#
#
# tot_zeros = np.sum(x == 0)
# tot_ones = np.sum(x == 1)
# print(f"{x=}")
# print(f"{tot_zeros%2=}")
# print(f"{tot_ones%2=}")
# print(f"{h[-1]=}")
# print()

print(f"[{', '.join(str(x_t) for x_t in x)}]")
print(f"[{', '.join(str(h_t) for h_t in h)}]")