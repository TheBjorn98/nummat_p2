import numpy as np

from support import concatflat, reconstruct_flat

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


def make_padding_function(paddingmode, d):
    if paddingmode == "zeros":
        def pad_zero(Y):
            d0, I = Y.shape
            Y_pad = np.zeros((d, I))
            Y_pad[:d0, :] = Y
            return Y_pad
        return pad_zero
    elif paddingmode == "tiling":
        def pad_tiling(Y):
            d0, I = Y.shape
            d_nice = np.ceil(d / d0) * d0
            Y_pad = np.tile(Y, (int(d_nice / d0), 1))
            return Y_pad[:d, :]
        return pad_tiling
    elif paddingmode == "repeat":
        def pad_repeat(Y):
            d0, I = Y.shape
            d_nice = np.ceil(d / d0) * d0
            Y_pad = np.repeat(Y, int(d_nice / d0), axis=0)
            return Y_pad[:d, :]
        return pad_repeat
    else:
        raise Exception("Padding mode not implemented!")


# Function to make function
def make_model_function(K, h, W, b, w, mu, pad_func):
    def model_function(Y):
        Y = pad_func(Y)
        d, I = Y.shape
        Z = Y

        for i in range(K):
            Z = Z + (
                h * sigma(
                    (W[:, :, i] @ Z).T + b[:, i]
                ).T
            )

        return np.squeeze(getUpsilon(Z, w, mu))
    return model_function


# Equation 4
# Returns all the hidden layer node values for I input vectors
# Y: d x I matrix of I input vectors
# K: Integer number of hidden layers
# h: stepsize
# W: d x d x K 3-tensor: colletion of weight matrices
# b: d x K matrix: list of all b_k offsets
def getZ(Y, K, h, W, b):
    # d = np.shape(Y)[0]
    # I = np.shape(Y)[1]
    d, I = Y.shape

    # Zs is a collection of K + 1 matrices
    # Each Z is the k-th set of intermediate values
    Zs = np.zeros((d, I, K + 1))

    Zs[:, :, 0] = Y

    for i in range(K):
        # TODO: Consider fixing indices so that transpose are no longer
        # necessary
        # or TODO: Explain transpose shit
        Zs[:, :, i + 1] = (
            Zs[:, :, i] + (
                h * sigma(
                    (W[:, :, i] @ Zs[:, :, i]).T + b[:, i]
                ).T
            )
        )

    return Zs


# Equation 5
# Returns d x I matrix of all the I Upsilon vectors
# ZK: d x I matrix of last layer Z
# mu: offset scalar
def getUpsilon(ZK, w, mu):
    return eta((ZK.T @ w) + mu)


# Equation 10
# Yc = (Upsilon - c)
def getPK(Yc, ZK, w, mu):

    # I = np.shape(ZK)[1]

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


# Equation 11
def getP(PK, Zs, h, K, W, b):

    d = np.shape(PK)[0]
    I = np.shape(PK)[1]

    Ps = np.zeros((d, I, K))

    Ps[:, :, K - 1] = PK

    for i in range(K - 2, 0, -1):
        # Ps[:, :, i] = (
        #     Ps[:, :, i + 1] + (
        #         h * W[:, :, i].T @ (
        #             sigPr(W[:, :, i] @ Zs[:, :, i] + b[:, i])
        #               * Ps[:, :, i + 1]
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


# Upsilon: I dim vector
# c: I vector of fasit
def getYc(Upsilon, c):
    return (Upsilon.T - c).T

# the non transpose of the non (Upsilon - c) part of equation 8
# This showed up a lot and was useful to have as a function


def getNu(ZK, w, mu):
    return etaPr(ZK.T @ w + mu)

# Common part between equation 12 and 13


def getHk(Ps, Zs, K, h, W, b):

    d = np.shape(Ps)[0]
    I = np.shape(Ps)[1]
    Hs = np.zeros((d, I, K))

    for i in range(K - 1):

        # TODO: recondense this expansion

        tmp1 = W[:, :, i] @ Zs[:, :, i]
        tmp2 = (tmp1.T + b[:, i]).T
        tmp3 = sigPr(tmp2)
        tmp4 = Ps[:, :, i + 1] * tmp3
        Hs[:, :, i] = h * tmp4
        # Hs[:, :, i] =
        #   h * (Ps[:, :, i + 1] * ((W[:, :, i] @ Zs[:, :, i]).T + b[:, i]).T)

        # print("Creating Hs[:, :, {}]:".format(i))
        # print("\nW @ Zk:\n{}".format(tmp1))
        # print("\nW @ Zk + bk:\n{}".format(tmp2))
        # print("\nsigPr(W @ Zk + bk):\n{}".format(tmp3))
        # print("\nP(k+1):\n{}".format(Ps[:, :, i + 1]))
        # print("\nP(k+1) o sigPr(W @ Zk + bk):\n{}".format(tmp4))

    return Hs


# Ups: d x I matrix of approx results
# c: I vector of exact results
# Ps: d x I x K tensor of all the "propagator values"
# Zs: d x I (K + 1) tensor of all the hidden layer node values
# h: step size
# W: d x d x K tensor of weights
# b: d x K matrix of offsets
# w: d vector of output weights
# mu: offset scalar
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
        dJdbk[:, i] = Hs[:, :, i] @ np.ones(I)

    # print("Hs:\n{}".format(Hs))
    # print("Zs:\n{}".format(Zs))

    # Should be d-vec
    # print("Hk: {}".format(np.shape(Hs[:, :, 0])))
    # dJdbk = np.array([ for i in range(K)])

    # dJdWk: d x d x K tensor of allllll the derivatives
    # dJdbk: d x K matrix of all the derivatives
    # dJdw: d vec of all derivatives
    # dJdmu: scalar derivative
    return (dJdWk, dJdbk, dJdw, dJdmu)


# tau: scalar learing factor (gradient decent step size)
# dJ: (3 dim, 2 dim, 1 dim, 0 dim) 4 tuple of derivatives
# W: hidden weights
# b: hidden offsets
# w: output weights
# mu: output offset
def updateTheta(tau, dJ, W, b, w, mu):

    Wn = W - tau * dJ[0]
    bn = b - tau * dJ[1]
    wn = w - tau * dJ[2]
    mun = mu - tau * dJ[3]

    return (tau, Wn, bn, wn, mun)


def setupAdam(dim):
    v = np.zeros(dim)
    m = np.zeros(dim)
    return (v, m, 1)


# TODO: find a concise way to construct theta from W, b, w and mu
# and recover these from theta
def adamTheta(state, dJ, W, b, w, mu):
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.01
    epsilon = 1e-8
    #
    v, m, i = state
    #
    shapes = [W.shape, b.shape, w.shape, np.array(mu).shape]
    theta = concatflat((W, b, w, mu))
    g = concatflat(dJ)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g * g)
    m_hat = m / (1 - beta1**i)
    v_hat = v / (1 - beta2**i)
    theta = theta + alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    #
    W, b, w, mu = reconstruct_flat(shapes, theta)
    return ((v, m, i + 1), W, b, w, mu)


def getGradANN(Y, K, h, W, b, w, mu):
    Zs = getZ(Y, K, h, W, b)
    acc = w * (etaPr(((Zs[:, :, K].T @ w).T + mu).T)).T
    # acc starts as grad(eta(ZK.T @ w + mu))

    for k in range(K, 0, -1):
        dphi = h * sigPr(((W[:, :, k - 1] @ Zs[:, :, k - 1]).T + b[:, k - 1]).T)
        acc = acc + (W[:, :, k - 1].T @ (dphi * acc))

    return acc  # acc is now grad_y(F) for (F is ANN)


# d: height of hidden layers
# K: amount of hidden layers
# h: stepsize
# tau: learining factor
# Y: d x I matrix of training input data
# c: I vector of traning answers
#
# TODO: Take padding mode and descent_mode/tau as parameters in some other way
def trainANN(
        d,
        K,
        h,
        Y,
        c,
        it_max,
        tol,
        tau=None,
        descent_mode="gradient",
        log=False):
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

    # Padding the input data

    d0, I = np.shape(Y)

    if d0 != d:
        raise Exception("Input must be padded")

    # print("Y = {}".format(Y))

    # Initialization

    W = np.random.uniform(size=(d, d, K))
    b = np.random.uniform(size=(d, K))
    w = np.random.uniform(size=(d, 1))
    mu = np.random.uniform(size=(1, 1))

    if descent_mode == "gradient":
        if tau is None:
            raise Exception("Must specify tau when using gradient descent!")
        descent_state = tau
        descent_function = updateTheta
    elif descent_mode == "adam":
        dim = np.sum([np.prod(W.shape), np.prod(b.shape), np.prod(w.shape), 1])
        descent_state = setupAdam(dim)
        descent_function = adamTheta

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
        # W, b, w, mu = updateTheta(.2, dJ, W, b, w, mu)
        (descent_state, W, b, w, mu) = descent_function(
            descent_state, dJ, W, b, w, mu)

        # Another iteration complete!
        it += 1

        if log:
            print(f"{it} / {it_max}, order of error: {np.log(J):.3f}")

    return (W, b, w, mu, Js)


# TODO: Make better name or something

def train_ANN_and_make_model_function(
        Y,
        c,
        d,
        K,
        h,
        it_max,
        tol,
        tau=None,
        descent_mode="gradient",
        padding_mode="zeros",
        activation_function=(sigma, sigPr),
        hypothesis_function=(eta, etaPr),
        log=False):
    # TODO:
    # - scale input Y
    # - scale output c
    # - train network on scaled values
    # - make model function
    # - make scaled model function where inputs are scaled by the same
    # factors as the training input, then the model function is called,
    # then the output is inversly scaled with the factors used to scale c
    # - make everything be able to take the activation and hypothesis
    # functions as parameters

    # Values to scale between
    alpha, beta = 0.2, 0.8

    # Save the min and max y values for scaling both the training data
    # and the input of the resulting model function
    y_min, y_max = np.min(Y), np.max(Y)

    # Scale the training data input
    Y = 1 / (y_max - y_min) * ((y_max - Y) * alpha + (Y - y_min) * beta)

    # Save the min and max training output data for scaling the training data
    # and inverse scaling the output of the model function
    c_min, c_max = np.min(c), np.max(c)

    # Scale the training data output
    c = 1 / (c_max - c_min) * ((c_max - c) * alpha + (c - c_min) * beta)

    d0, I = Y.shape

    # Set the padding function and pad the input data
    if d0 < d:
        pad_func = make_padding_function(padding_mode, d)
        Y = pad_func(Y)
    elif d0 > d:
        raise Exception(
            "Dimension of input is larger than"
            + " dimension of neural net!")
    else:
        def identity(y):
            return y
        pad_func = identity

    (W, b, w, mu, Js) = trainANN(d, K, h, Y, c,
                                 it_max, tol, tau=tau,
                                 descent_mode=descent_mode, log=log)

    modfunc = make_model_function(K, h, W, b, w, mu, pad_func)

    def scaled_modfunc(Y):
        # The padding functions do not like getting one dimensional input
        # but we want to be able to use our model function as a function
        # of a single vector, or a whole matrix of vector, so in case it's
        # a vector, it is simply reshaped to a matrix with a single column
        # or in the very special case its a single number, it is reshaped
        # to a 1x1 matrix
        if isinstance(Y, float):
            Y = np.reshape(Y, (1, 1))
        elif len(Y.shape) == 1:
            Y = np.reshape(Y, (len(Y), 1))

        # Scale the input
        Y_scaled = 1 / (y_max - y_min) * ((y_max - Y) * alpha + (Y - y_min) * beta)

        # Compute the scaled output
        c_scaled = modfunc(Y_scaled)

        # Apply the inverse scaling to the output
        c = 1 / (beta - alpha) * ((beta - c_scaled)
                                  * c_min + (c_scaled - alpha) * c_max)

        return c

    def gradient_modfunc(Y):
        # The padding functions do not like getting one dimensional input
        # but we want to be able to use our model function as a function
        # of a single vector, or a whole matrix of vector, so in case it's
        # a vector, it is simply reshaped to a matrix with a single column
        # or in the very special case its a single number, it is reshaped
        # to a 1x1 matrix
        if isinstance(Y, float):
            Y = np.reshape(Y, (1, 1))
        elif len(Y.shape) == 1:
            Y = np.reshape(Y, (len(Y), 1))

        Y = pad_func(Y)

        Y_scaled = 1 / (y_max - y_min) * ((y_max - Y) * alpha + (Y - y_min) * beta)

        # gradent of the scaled c with respect to the scaled input Y
        dc_scaled_scaled = getGradANN(Y_scaled, K, h, W, b, w, mu)

        # gradient of the scaled c with respect to the unscaled input Y
        scale_factor = (c_max - c_min) / (y_max - y_min)
        dc = dc_scaled_scaled  # * scale_factor
        print(f"Scaling with: {scale_factor:.5f}")

        return dc

    # Return the scaled model function and the evolution of the
    # objective function for analisys
    return (scaled_modfunc, gradient_modfunc, Js)
