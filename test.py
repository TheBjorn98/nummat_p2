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

    d, I = 4, 10
    K, h = 5, .2
    it_max, tol = 1000, 1e-4
    tau = .5

    # _Y = np.random.uniform(low=-2., high=2., size=(d, I))
    # _Y = np.sort(_Y)
    _Y = np.array([np.linspace(-1, 1, num=I), np.ones(I)])
    Y = support.scaleInput(_Y, .2, .8)
    # TODO: fix the test function so it does this automagically
    c = np.array([F(y) for y in Y.T])
    print(c)

    testType = "trainANN"

    if testType == "trainANN":

        W, b, w, mu, Js = ann.trainANN(d, K, h, tau, Y, c, it_max, tol)

        Z_fin = ann.getZ(Y, K, h, W, b)
        Ups = ann.getUpsilon(Z_fin[:, :, -1], w, mu)
        # print("Analytical: \n{}".format(c))
        # print("{} updates: \n{}".format(it_max, Ups))
        plt.plot(Js)
        plt.show()
        plt.plot(c)
        plt.plot(Ups)
        plt.show()

        '''
        grad = ann.getGradANN(Y, K, h, W, b, w, mu)
        # print(Y)
        # print(grad)
        print((Y - grad) / Y * 100)
        for y in Y:
            plt.plot(np.sort(y))
        for g in grad:
            plt.plot(np.sort(g))

        plt.show()
        '''

    if testType == "fixWs":

        W = np.random.uniform(size=(d, d, K))
        b = np.random.uniform(size=(d, K))
        w = np.random.uniform(size=(d, 1))
        mu = np.random.uniform(size=(1, 1))

        Zs = ann.getZ(Y, K, h, W, b)
        Ups = ann.getUpsilon(Zs[:, :, -1], w, mu)
        Yc = (Ups.T - c).T
        PK = ann.getPK(Yc, Zs[:, :, -1], w, mu)
        # print("PK: {}".format(np.shape(PK)))
        Ps = ann.getP(PK, Zs, h, K, W, b)
        dJ = ann.getdelJ(Ups, c, Ps, Zs, K, h, W, b, w, mu)

        Wn, bn, wn, mun = ann.updateTheta(1, dJ, W, b, w, mu)

        # print("dJdWk: {}".format(np.shape(dJ[0])))
        # print("dJdbk: {}".format(np.shape(dJ[1])))
        # print("dJdw: {}".format(np.shape(dJ[2])))
        # print("dJdmu: {}".format(np.shape(dJ[3])))

        # print("Y: \n{}\n".format(Y))

        print("dJdW: \n{}".format(dJ[0]))
        # print("Wn: \n{}".format(Wn))

        print("W: \n{}".format(W))
        print("Wn: \n{}".format(Wn))
        # print("bn: \n{}".format(bn))

        Z2 = ann.getZ(Y, K, h, Wn, bn)
        U2 = ann.getUpsilon(Z2[:, :, -1], wn, mun)

        print("Analytical: \n{}".format(c))
        print("Random init: \n{}".format(Ups))
        # print("One update: \n{}".format(U2))
