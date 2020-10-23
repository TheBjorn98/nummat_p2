import numpy as np
import matplotlib.pyplot as plt

import time

import ann


def make_uniform_inp_box(low, high, n):
    return np.random.uniform(low, high, (n, len(low))).T


def testfunc_1(y):
    return 0.5 * (y[0]**2 + y[1]**2)


def testfunc_2(y):
    return 1.0 - np.cos(y)
    # return np.sin(y) * y


def train_test_func1():
    I = 100
    Y = make_uniform_inp_box([-2, -2], [2, 2], I)
    c = np.array([testfunc_1(Y[:, i]) for i in range(I)])

    # print("Y = {}".format(Y.T))
    # print("c = {}".format(c))

    tau = 0.2
    d = 8
    K = 4
    h = 0.1
    it_max = 1000
    tol = 1e-4

    t = time.time()

    (W, b, w, mu, _) = ann.trainANN(d, K, h, Y, c, it_max, tol, tau=tau)

    f = ann.make_model_function(K, h, W, b, w, mu, "repeat", d)

    testy = np.array([[0.5, 1.0]]).T

    print(f(testy))
    print(testfunc_1(testy))

    # print("took {} seconds".format(time.time() - t))


def train_test_func2():
    I = 100
    Y = make_uniform_inp_box([-np.pi / 3], [np.pi / 3], I)
    # print(Y)
    c = np.squeeze(np.array([testfunc_2(Y[:, i]) for i in range(I)]))
    # print(c)

    # print("Y = {}".format(Y.T))
    # print("c = {}".format(c))

    tau = 0.1
    d = 2
    K = 8
    h = 0.1
    it_max = 10000
    tol = 1e-4

    t = time.time()

    (f, Js) = ann.train_ANN_and_make_model_function(
        Y, c, d, K, h, it_max, tol, tau=tau)
    
    print(f"took {time.time() - t:.3f} seconds")
    
    plt.plot(np.log(Js))
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

    # print("took {} seconds".format(time.time() - t))


# TODO: Do scaling for input and output values, and inverse scaling for
# the actual use of the methods
# TODO: Fix dimension problems when fitting one dimensional functions


train_test_func2()
