import itertools
import matplotlib.pyplot as plt
import numpy as np

from time import time

import ann

import sys

def f1(y):
    return 1 / 2 * y**2


def df1(y):
    return y


def f2(y):
    return 1 - np.cos(y)


def df2(y):
    return np.sin(y)


def f3(y):
    return y**3 - y


def df3(y):
    return 3 * y**2 - 1


functions_1d = [(f1, df1, [-2, 2]),
                (f2, df2, [-np.pi / 3, np.pi / 3]), (f3, df3, [-1, 1])]


def train_1d_function(i, I, d, K, h, tau, it_max, tol, filename=None):
    f, df, interval = functions_1d[i]

    Y = np.linspace(interval[0], interval[1], I)

    c = f(Y)

    Y = np.reshape(Y, (1, I))

    t = time()

    (F, dF, Js, _) = ann.train_ANN_and_make_model_function(
        Y, c, d, K, h, it_max, tol, tau=tau, padding_mode="zeros")

    print(f"took {time() - t:.2f} seconds to train")

    print(f"J = {Js[-1]}")

    plt.plot(Js)
    plt.yscale("log")
    if filename is not None:
        plt.savefig("J.png")
        plt.savefig("J.pdf")
        plt.clf()
        plt.cla()
    else:
        plt.show()

    n = 1000
    y_plot_grid = np.linspace(interval[0], interval[1], n)

    z = f(y_plot_grid)
    Z = F(np.reshape(y_plot_grid, (1, n)))

    dz = df(y_plot_grid)
    dZ = dF(np.reshape(y_plot_grid, (1, n)))

    plt.plot(y_plot_grid, z, label="f")
    plt.plot(y_plot_grid, Z, label="F")
    plt.legend()
    if filename is not None:
        plt.savefig("F.png")
        plt.savefig("F.pdf")
        plt.clf()
        plt.cla()
    else:
        plt.show()

    plt.plot(y_plot_grid, dz, label="df")
    plt.plot(y_plot_grid, dZ, label="dF")
    plt.legend()
    if filename is not None:
        plt.savefig("dF.png")
        plt.savefig("dF.pdf")
        plt.clf()
        plt.cla()
    else:
        plt.show()


# train_1d_function(0, 10, 4, 4, 1, 0.1, 10000, 1e-4)


def make_multid_grid(low, high, n):
    spaces = [np.linspace(l, h, n) for (l, h) in zip(low, high)]
    dV = np.prod([s[1] - s[0] for s in spaces])
    return np.array([np.array(x) for x in itertools.product(*spaces)]).T, dV


def normify_grad_matrix(G):
    return np.array([np.linalg.norm(v) for v in G.T])


def f2d1(Y):
    return 1 / 2 * (Y[0, :]**2 + Y[1, :]**2)


def df2d1(Y):
    return Y


def f2d2(Y):
    return -1 / np.sqrt(Y[0, :]**2 + Y[1, :]**2)


def df2d2(Y):
    return np.array([Y[0, :], Y[1, :]]) / (Y[0, :]**2 + Y[1, :]**2)**(3 / 2)


def f3d1(Y):
    return 1 / 2 * (Y[0, :]**2 + Y[1, :]**2 + Y[2, :]**2)


def df3d1(Y):
    return Y


functions_nd = [(f2d1, df2d1, [(-2, -2), (2, 2)]),
                (f2d2, df2d2, [(-2, -2), (2, 2)]),
                (f3d1, df3d1, [(-2, -2, -2), (2, 2, 2)])]


def train_nd_function(i, I, d, K, h, tau, it_max, tol, filename=None):
    f, df, intervals = functions_nd[i]

    Y, _ = make_multid_grid(intervals[0], intervals[1], I)

    c = f(Y)

    t = time()

    (F, dF, Js, _) = ann.train_ANN_and_make_model_function(
        Y, c, d, K, h, it_max, tol, tau=tau, padding_mode="zeros", log="log.txt")

    print(f"took {time() - t:.2f} seconds to train")

    print(f"J = {Js[-1]}")

    plt.plot(Js)
    plt.yscale("log")
    if filename is not None:
        plt.savefig("J.png")
        plt.savefig("J.pdf")
        plt.clf()
        plt.cla()
    else:
        plt.show()

    Y_eval_grid, dV = make_multid_grid(intervals[0], intervals[1], 50)

    z = f(Y_eval_grid)
    Z = F(Y_eval_grid)
    e = np.sum((z - Z)**2) * dV
    print(f"function error: {e}")

    dz = df(Y_eval_grid)
    dZ = dF(Y_eval_grid)
    e = np.sum(normify_grad_matrix(dz - dZ)) * dV
    print(f"gradient error: {e}")


# train_nd_function(2, 8, 4, 4, 1, 0.005, 10000, 1e-4)

if __name__ == "__main__":
    argv = sys.argv
    function_dim = int(argv[1])
    i = int(argv[2])
    I = int(argv[3])
    d = int(argv[4])
    K = int(argv[5])
    h = float(argv[6])
    tau = float(argv[7])
    it_max = int(argv[8])
    tol = float(argv[9])
    filename = argv[10]
    print(f"""
Job Info:
function_dim = {function_dim}
i = {i}
I = {I}
d = {d}
K = {K}
h = {h}
tau = {tau}
it_max = {it_max}
tol = {tol}
filename = {filename}
""")
    if function_dim == 1:
        train_1d_function(i, I, d, K, h, tau, it_max, tol, filename=filename)
    else:
        train_nd_function(i, I, d, K, h, tau, it_max, tol, filename=filename)
