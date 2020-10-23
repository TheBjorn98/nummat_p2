import ann
import numpy as np
import support
import matplotlib.pyplot as plt


def F(y):
    return 1 / 2 * np.sum(y ** 2)


if __name__ == "__main__":

    # Problem dimension: d = 2
    # Independent test data: I = 10
    # Number of hidden layers: K = 3
    # Stepsize for internal layers: h = 1

    d, I = 2, 100
    K, h = 3, .15
    it_max, tol = 10000, 1e-4
    tau = .2

    # _Y = np.random.uniform(low=-2., high=2., size=(d, I))
    # _Y = np.sort(_Y)
    _Y = np.array([np.linspace(-1, 1, num=I), np.ones(I)])
    Y = support.scaleInput(_Y, .2, .8)
    # TODO: fix the test function so it does this automagically
    c = np.array([F(y) for y in Y.T])
    print(c)

    # (W, b, w, mu, Js) = ann.trainANN(d, K, h, Y, c, it_max, tol, tau=tau)

    f, Js = ann.train_ANN_and_make_model_function(Y, c, d, K, h, it_max, tol, tau=tau)

    x = np.linspace(.1, .9, num=I)
    y = np.ones(I)

    f_p = f(np.array([x, y]))

    plt.plot(Js)
    plt.yscale("log")
    plt.show()
    plt.plot(c)
    plt.plot(f_p)
    plt.show()
