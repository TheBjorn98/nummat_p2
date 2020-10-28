import numpy as np


def f(x):
    return x[0, :]**2 + x[1, :]**3


def df(x):
    return np.array([x[0, :]**2, x[1, :]**2])


xs = np.array([
    [0, 0.0],
    [1, 1],
    [1, 2]
]).T

print(xs)


print(f(xs))

print(df(xs))
