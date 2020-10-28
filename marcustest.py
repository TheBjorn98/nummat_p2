import numpy as np
import matplotlib.pyplot as plt

import time

import ann


def make_uniform_inp_box(low, high, n):
    return np.random.uniform(low, high, (n, len(low))).T


def testfunc_1(y):
    return 0.5 * (y[0]**2 + y[1]**2)


def testfunc_2(y):
    # return 1.0 - np.cos(y)
    # return y**3 - y
    return y**4 - y**2
    # return np.sin(y) * y


def testfunc_1_2(x, y):
    return testfunc_1([x, y])


# def d_testfunc_2(y):
#     # return np.sin(y)
#     # return 3 * y**2

def plot_2d(X, Y, Z):
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    # plt.savefig("tmp.pdf")
    plt.show()


def train_test_func1():
    # I = 100
    n = 20
    I = n**2
    # Y = make_uniform_inp_box([-2, -2], [2, 2], I)
    Y = np.reshape(np.meshgrid(np.linspace(-2, 2, n),
                               np.linspace(-2, 2, n)), (2, I))

    # print(f"Y = {Y}")
    c = np.array([testfunc_1(Y[:, i]) for i in range(I)])
    # print(f"c = {c}")

    # return

    tau = 0.05
    d = 4
    K = 4
    h = 1
    it_max = 10000
    tol = 1e-4

    t = time.time()

    (f, df, Js) = ann.train_ANN_and_make_model_function(
        Y, c, d, K, h, it_max, tol, tau=tau, padding_mode="repeat")

    print(f"took {time.time() - t:.3f} seconds")

    plt.plot(np.log10(Js))
    plt.show()

    # print("took {} seconds".format(time.time() - t))


def train_test_func2():
    I = 100
    # Y = make_uniform_inp_box([-np.pi / 3], [np.pi / 3], I)
    Y = np.reshape(np.linspace(-np.pi / 3, np.pi / 3, I), (1, I))
    # print(Y)
    c = np.squeeze(np.array([testfunc_2(Y[:, i]) for i in range(I)]))
    # print(c)

    print("Y = {}".format(Y.T))
    # print("c = {}".format(c))

    tau = 0.1
    d = 4
    K = 6
    h = 0.1
    it_max = 10000
    tol = 1e-4

    t = time.time()

    (f, df, Js) = ann.train_ANN_and_make_model_function(
        Y, c, d, K, h, it_max, tol, tau=tau, padding_mode="repeat")

    print(f"took {time.time() - t:.3f} seconds")

    plt.plot(np.log10(Js))
    # plt.show()
    plt.savefig("js.png")
    plt.clf()
    plt.cla()

    n = 100
    ys = np.linspace(-np.pi / 3, np.pi / 3, n)

    cs = f(np.reshape(ys, (1, n)))

    plt.plot(ys, testfunc_2(ys))
    plt.plot(ys, cs)
    plt.savefig("func_fit.png")
    plt.clf()
    plt.cla()

    # dcs = df(np.reshape(ys, (1, n)))[0, :]

    # plt.plot(ys, d_testfunc_2(ys))
    # plt.plot(ys, dcs)
    # plt.savefig("d_func_fit.png")
    # plt.clf()
    # plt.cla()

    # print("took {} seconds".format(time.time() - t))


# TODO: Do scaling for input and output values, and inverse scaling for
# the actual use of the methods
# TODO: Fix dimension problems when fitting one dimensional functions


train_test_func2()
