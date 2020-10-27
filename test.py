import ann
import numpy as np
import matplotlib.pyplot as plt
import time


# def F(y):
#     return 1 / 2 * np.sum(y ** 2)

def testfunc_2(y):
    return 1.0 - np.cos(y)
    # return np.sin(y) * y
    # return y**2 / 2


def dtestfunc_2(y):
    return np.sin(y)
    # return np.cos(y) * y + np.sin(y)
    # return y


if __name__ == "__main__":

    print("Hello there")

    # Problem dimension: d = 2
    # Independent test data: I = 10
    # Number of hidden layers: K = 3
    # Stepsize for internal layers: h = 1

    I = 10
    Y = np.reshape(np.linspace(-np.pi / 3, np.pi / 3, I), (1, I))
    c = np.squeeze(np.array([testfunc_2(Y[:, i]) for i in range(I)]))

    tau = 0.5
    d = 4
    K = 16
    h = 0.1
    it_max = 10000
    tol = 1e-4

    t = time.time()

    (f, df, Js) = ann.train_ANN_and_make_model_function(
        Y, c, d, K, h, it_max, tol, tau=tau, padding_mode="repeat", log=False)

    print(f"Took {time.time() - t:.3f} seconds, {len(Js)} iterations")

    n = 100
    ys = np.linspace(-np.pi / 3, np.pi / 3, n)
    # ys = np.reshape(np.linspace(-np.pi / 3, np.pi / 3, n), (1, n))

    plotFunc = True
    plotGrad = True
    plotErr = True

    if plotFunc:
        cs = f(np.reshape(ys, (1, n)))
        plt.plot(ys, testfunc_2(ys))
        plt.plot(ys, cs)
        plt.show()

    if plotGrad:
        dcs = df(np.reshape(ys, (1, n)))[0, :]
        plt.plot(ys, dtestfunc_2(ys))
        plt.plot(ys, dcs)
        plt.show()

    if plotErr:
        plt.plot(Js)
        plt.show()

    # grad_ann = df(ys)
    # print(grad_ann)
    # print(grad_ann)
    # grad_exact = np.sin(ys)

    # plt.plot(ys, grad_exact)
    # plt.plot(ys, grad_ann)
    # plt.show()
