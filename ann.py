import numpy as np

# Script defining the necessary functions for the ANN

# TODO: add sig, sigPr, eta and etaPr as inputs to functions
# TODO: change variable names to be more descriptive
# both of these are followed by a question mark, as "idk if should do, lol"


# Activation function, sigmoid function
def sigma(x):
    return np.tanh(x)


# Derivative of activation function wrt. x
# Used for back-propagation
def sigPr(x):
    return 1 - np.tanh(x)**2


# Hypothesis function, identity function
def eta(x):
    return .5 * (1 + np.tanh(x / 2))


# Derivative of hypothesis function wrt. x
# Used for back-propagation in the step from output to output layer
def etaPr(x):
    return .25 * (1 - np.tanh(x / 2)**2)


def getZ(Y, K, h, W, b):
    d = np.shape(Y)[0]
    I = np.shape(Y)[1]

    # Zs is a collection of K + 1 matrices
    # Each Z is the k-th set of intermediate values
    Zs = np.zeros((d, I, K + 1))

    Zs[:, :, 0] = Y

    for i in range(1, K + 1):
        # TODO: Consider fixing indices so that transpose are no longer
        # necessary
        Zs[:, :, i] = (
            Zs[:, :, i - 1] + (
                h * sigma(
                    (W[:, :, i - 1] @ Zs[:, :, i - 1]).T + b[:, i - 1]
                ).T
            )
        )

    return Zs


def getUpsilon(ZK, w, mu):
    return eta((ZK.T @ w) + mu)


def getPK(Yc, ZK, w, mu):

    I = np.shape(ZK)[1]

    # t1 = ((ZK.T @ w).T + mu).T
    # t2 = etaPr(t1)
    # t3 = Yc * t2
    # print("ZK: {}".format(np.shape(ZK)))
    # print("w: {}".format(np.shape(w)))
    # print("ZK.T @ w + mu: {}".format(np.shape(t1)))
    # print("etaPr: {}".format(np.shape(t2)))
    # print("Yc: {}".format(np.shape(Yc)))
    # print("Yc * etaPr(ZK.T @ w + mu): {}".format(np.shape(t3)))
    # t4 = np.outer(w, t3.T)

    # return t4

    return np.outer(
        w, (
            Yc * etaPr(
                ((ZK.T @ w).T + mu).T  # * np.ones((I, 1))
            )
        ).T
    )


def getP(PK, Zs, h, K, W, b):

    d = np.shape(PK)[0]
    I = np.shape(PK)[1]

    Ps = np.zeros((d, I, K))

    Ps[:, :, K - 1] = PK

    for i in range(K - 2, 0, -1):
        # Ps[:, :, i] = (
        #     Ps[:, :, i + 1] + (
        #         h * W[:, :, i].T @ (
        #             sigPr(W[:, :, i] @ Zs[:, :, i] + b[:, i]) * Ps[:, :, i + 1]
        #         )
        #     )
        # )

        # TODO: recondense this expansion

        tmp1 = W[:, :, i] @ Zs[:, :, i]
        tmp2 = (tmp1.T + b[:, i]).T
        tmp3 = sigPr(tmp2)
        tmp4 = Ps[:, :, i + 1] * tmp3
        tmp5 = W[:, :, i].T @ tmp4
        Ps[:, :, i] = Ps[:, :, i + 1] + h * tmp5

        # print("Creating Ps[:, :, {}]:".format(i))
        # print("\nW @ Zk-1:\n{}".format(tmp1))
        # print("\nW @ Zk-1 + bk-1:\n{}".format(tmp2))
        # print("\nsigPr(W @ Zk-1 + bk-1):\n{}".format(tmp3))
        # print("\nP(k):\n{}".format(Ps[:, :, i + 1]))
        # print("\nP(k) o sigPr(W @ Zk-1 + bk-1):\n{}".format(tmp4))
        # print("\nW.T @ P(k) o sigPr(W @ Zk-1 + bk-1):\n{}".format(tmp5))
        # print("\nResult of P{}:\n{}".format(i, Ps[:, :, i]))

    return Ps


def getYc(Upsilon, c):
    return (Upsilon.T - c).T


def getNu(ZK, w, mu):
    return etaPr(ZK.T @ w + mu)


def getHk(Ps, Zs, K, h, W, b):

    d = np.shape(Ps)[0]
    I = np.shape(Ps)[1]
    Hs = np.zeros((d, I, K))

    for i in range(K-1):

        # TODO: recondense this expansion

        tmp1 = W[:, :, i] @ Zs[:, :, i]
        tmp2 = (tmp1.T + b[:, i]).T
        tmp3 = sigPr(tmp2)
        tmp4 = Ps[:, :, i + 1] * tmp3
        Hs[:, :, i] = h * tmp4
        # Hs[:, :, i] = h * (Ps[:, :, i + 1] * ((W[:, :, i] @ Zs[:, :, i]).T + b[:, i]).T)

        # print("Creating Hs[:, :, {}]:".format(i))
        # print("\nW @ Zk:\n{}".format(tmp1))
        # print("\nW @ Zk + bk:\n{}".format(tmp2))
        # print("\nsigPr(W @ Zk + bk):\n{}".format(tmp3))
        # print("\nP(k+1):\n{}".format(Ps[:, :, i + 1]))
        # print("\nP(k+1) o sigPr(W @ Zk + bk):\n{}".format(tmp4))

    return Hs


def getdelJ(Ups, c, Ps, Zs, K, h, W, b, w, mu):

    d = np.shape(Ps)[0]
    I = np.shape(Ps)[1]

    Yc = getYc(Ups, c)
    nu = getNu(Zs[:, :, K], w, mu)
    Hs = getHk(Ps, Zs, K, h, W, b)

    dJdmu = nu.T @ Yc  # Should be scalar
    dJdw = Zs[:, :, K] @ (Yc * nu)  # Should be d-vec

    # Should be dxd mx
    dJdWk = np.zeros((d, d, K))
    dJdbk = np.zeros((d, K))
    for i in range(K):
        dJdWk[:, :, i] = Hs[:, :, i] @ Zs[:, :, i].T
        dJdbk[:, i] = Hs[:, :, i] @ np.ones((I))

    # print("Hs:\n{}".format(Hs))
    # print("Zs:\n{}".format(Zs))

    # Should be d-vec
    # print("Hk: {}".format(np.shape(Hs[:, :, 0])))
    # dJdbk = np.array([ for i in range(K)])

    return (dJdWk, dJdbk, dJdw, dJdmu)


def updateTheta(tau, dJ, W, b, w, mu):

    Wn = W - tau * dJ[0]
    bn = b - tau * dJ[1]
    wn = w - tau * dJ[2]
    mun = mu - tau * dJ[3]

    return (Wn, bn, wn, mun)


# TODO: find a concise way to construct theta from W, b, w and mu
# and recover these from theta
def adamTheta(dJ, W, b, w, mu):
    pass


def getGradANN(Y, K, h, W, b ,w, mu):
    Zs = getZ(Y, K, h, W, b)
    acc = w * etaPr(Zs[:, :, K])  # acc starts as grad(eta(Wk @ ZK + bk))

    for k in range(K, 0, -1):
        dphi = h * sigPr(Zs[:, :, k - 1])
        acc = acc + h * (dphi * acc)

    return acc  # acc is now grad_y(F) for (F is ANN)


def trainANN(d, K, h, tau, Y, c, it_max, tol):
    '''
    d, K, h and tau are model parameters for:
        d: dimension of spaces in hidden layers
        K: number of hidden layers in the ResNet model
        h: stepsize for emphazising internal layers in the model
        tau: learning parameter declaring how much of the gradient is included
    Y: d0xI matrix with input data, must be scaled beforehand
    c: I-vector of exact data for evaluating performance
    it_max: sets a cieling for compute time, maximal number of training rounds
    tol: error tolerance, when error dips below, model is done training

    Returns: Weights and activations for the ResNet
        W, b, w, mu
    '''

    # TODO: implement more sophisticated tolerance criterion

    d0, I = np.shape(Y)

    if d0 < d:
        pass
        # TODO: code for embedding data if the dimensions mismatch

    # Initialization

    W = np.random.uniform(size=(d, d, K))
    b = np.random.uniform(size=(d, K))
    w = np.random.uniform(size=(d, 1))
    mu = np.random.uniform(size=(1, 1))

    # Iteration

    it = 0
    J = np.inf
    Js = []

    while it < it_max and J > tol:

        # Computing Z's in the forward sweep
        Zs = getZ(Y, K, h, W, b)

        # Applying hypothesis function to recover approximate answer
        Ups = getUpsilon(Zs[:, :, -1], w, mu)

        # Getting diff. in exact and approx and evaluating objective func
        Yc = (Ups.T - c).T
        J = .5 * np.linalg.norm(Yc) ** 2
        Js.append(J)

        # Preparing for back-propagation, getting PK = dJ/dZK
        PK = getPK(Yc, Zs[:, :, -1], w, mu)
        # print("PK: {}".format(np.shape(PK)))

        # Getting all Pk for back-propagation
        Ps = getP(PK, Zs, h, K, W, b)
        # Computing gradient of obj.func. for use in update
        dJ = getdelJ(Ups, c, Ps, Zs, K, h, W, b, w, mu)

        # Updating the weights and activations in the model
        # Update-scheme followinmg from details in the project
        W, b, w, mu = updateTheta(.2, dJ, W, b, w, mu)

        # Another iteration complete!
        it += 1

        # if J < .1 * Js[0]:
        #     break

    return (W, b, w, mu, Js)
