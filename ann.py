import numpy as np
# import matplotlib.pyplot as plt
from time import time
from support import concatflat, reconstruct_flat

# Script defining the necessary functions for the ANN

# TODO: add sig, sigPr, eta and etaPr as inputs to functions
# TODO: change variable names to be more descriptive
# both of these are followed by a question mark, as "idk if should do, lol"


# Activation function, sigmoid function
def sigma(x):
    '''Activation function, sigmoid'''
    return np.tanh(x)


# Derivative of activation function wrt. x
# Used for back-propagation
def sigPr(x):
    '''Derivative of activation function'''
    return 1 - np.tanh(x)**2


# Hypothesis function, identity function
def eta(x):
    '''Hypothesis function'''
    return .5 * (1 + np.tanh(x / 2))


# Derivative of hypothesis function wrt. x
# Used for back-propagation in the step from output to output layer
def etaPr(x):
    '''Derivative of the hypothesis function'''
    return .25 * (1 - np.tanh(x / 2)**2)


def make_padding_function(paddingmode, d):
    '''
    Returns function which pads the input data as to embedd it in a larger
    vectorspace.

    zeros: Appends zeros until d is reached
        [1, 2] -> [1, 2, 0, 0]
    tiling: Appends the vector itself until d is reached, superfluous elements
    are dropped
        [1, 2] -> [1, 2, 1, 2]
    repeat: Appends elementwise so the first element is repeated a number of
    times, and so on
        [1, 2] -> [1, 1, 2, 2]
    '''
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


# Function to make function approximator as if it was a "normal function"
# Performs only a forward sweep and saves no intermediate steps to save RAM
def make_model_function(K, h, W, b, w, mu, pad_func):
    '''Input:
        (W, b, w, mu) = theta of the ANN
        K: number of hidden layers
        h: stepsize between layers
        pad_func: padding scheme
    Returns:
        model_function: Performing only a forward sweep, saving nothing
    '''
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
# Returns all intermediate values for I input vectors
# Y: d x I matrix of I input vectors
# K: Integer number of hidden layers
# h: stepsize
# W: d x d x K 3-tensor: colletion of weight matrices
# b: d x K matrix: list of all b_k offsets
def getZ(Y, K, h, W, b):
    '''Computes all Zk in a forward sweep

    Input:
        Y: input data (dxI matrix)
        K: number of hidden layers
        h: stepsize between layers
        W: weights between layers
        b: offsets between layers
    Forward sweep to get all Z^{(k)}

    Return:
        Zs: all intermediate values of the network at the
        current state in a (d, d, K+1) collection of matrices
    '''
    d, I = Y.shape

    # Zs is a collection of K + 1 matrices
    # Each Z is the k-th set of intermediate values
    Zs = np.zeros((d, I, K + 1))

    Zs[:, :, 0] = Y

    for i in range(K):
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
    '''Computes the output values from the function approximator
    by using the equation for the terminal layer

    Return: Upsilon
    '''
    return eta((ZK.T @ w) + mu)


# Equation 10
# Yc = (Upsilon - c)
def getPK(Yc, ZK, w, mu):
    '''
    Get first back propagation vector (or dxI matrix)

    Input:
        Yc: (Upsilon - c)
        ZK: Intermediate values at terminal layer
        w : Weights used in G(ZK)
        mu: Offset used in G(ZK)
    Return:
        PK: delJ / delZK, first set of propagation vectors
    '''
    return np.outer(
        w,
        (Yc * etaPr(((ZK.T @ w).T + mu).T)).T
    )


# Equation 11
def getP(PK, Zs, h, W, b):
    '''
    Computes all remaining back propagation vectors

    Input:
        PK: back propagation for terminal layer
        Zs: all Zk from the forward sweep
        h : stepsize between layers
        W : weights between hidden layers
        b : offset between hidden layers
    '''
    d, I = np.shape(PK)
    K = np.shape(W)[2]

    Ps = np.zeros((d, I, K))

    Ps[:, :, K - 1] = PK

    # The lines are split for readability
    # The calculations correspont to eq 11 in the project
    for i in range(K - 2, 0, -1):
        # sig_k: sigma'(Wk Zk + bk)
        sig_k = sigPr(((W[:, :, i] @ Zs[:, :, i]).T + b[:, i]).T)
        # hadamard: W_k^T (sig_k * Pk)
        hadamard = W[:, :, i].T @ (Ps[:, :, i + 1] * sig_k)
        # Pk = Pk-1 + h * scnd_term
        Ps[:, :, i] = Ps[:, :, i + 1] + h * hadamard

    return Ps


# Upsilon: I-vector
# c: I vector of exact values
def getYc(Upsilon, c):
    '''Computes Yc = Upsilon - c to save recomputations'''
    return (Upsilon.T - c).T


def getNu(ZK, w, mu):
    '''Computes nu = eta'(ZK^T w + mu) to save recomputations'''
    return etaPr(ZK.T @ w + mu)


def getHk(Ps, Zs, h, W, b):
    '''Computes Hk (common part between eq 12 and 13) to save recomputation'''
    d, I, K = np.shape(Ps)
    Hs = np.zeros((d, I, K))

    for i in range(K - 1):
        sig_k = sigPr(((W[:, :, i] @ Zs[:, :, i]).T + b[:, i]).T)
        hadamard = Ps[:, :, i + 1] * sig_k
        Hs[:, :, i] = h * hadamard

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
def getdelJ(Ups, c, Ps, Zs, h, W, b, w, mu):

    d, I, K = np.shape(Ps)

    Yc = getYc(Ups, c)
    nu = getNu(Zs[:, :, K], w, mu)
    Hs = getHk(Ps, Zs, h, W, b)

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
    '''Updates theta according to eq. 7 in the project description

    Input:
        tau: learning factor
        dJ : gradient of the objective function wrt. theta
        W, b, w, mu: theta, really
    Output:
        tau: to keep descent mode consistent and distinguish from ADAM
        Wn, bn, wn, mun: Updated theta
    '''
    Wn = W - tau * dJ[0]
    bn = b - tau * dJ[1]
    wn = w - tau * dJ[2]
    mun = mu - tau * dJ[3]

    return (tau, Wn, bn, wn, mun)


def setupAdam(dim):
    v = np.zeros(dim)
    m = np.zeros(dim)
    return (v, m, 1)


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


def getGradANN(Y, h, W, b, w, mu):
    '''Computes gradient according to derivation from the project report
    section 3 subsection "Implementation of the derived gradient

    Input:
        Y: dxI matrix of all input
        h: stepsize between layers
        W, b, w, mu: theta
    Output:
        acc: the accumulated gradient from backtracking through all layers
    '''
    K = np.shape(W)[2]
    Zs = getZ(Y, K, h, W, b)
    acc = w @ (etaPr(((Zs[:, :, K].T @ w).T + mu).T)).T
    # acc starts as grad(eta(ZK.T @ w + mu))

    for k in range(K, 0, -1):
        dphi = h * \
            sigPr(((W[:, :, k - 1] @ Zs[:, :, k - 1]).T + b[:, k - 1]).T)
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
        log=False,
        theta=None):
    '''Trains a ResNet ANN within the given parameters

    Input:
        d:   dimension of spaces in hidden layers
        K:   number of hidden layers in the ResNet model
        h:   stepsize for emphazising internal layers in the model
        Y: d0xI matrix with input data, must be scaled beforehand
        c: I-vector of exact data for evaluating performance
        it_max: maximal number of training rounds, ceiling for training time
        tol: error tolerance, when error dips below, model is done training
        tau: learning parameter declaring how much of the gradient is included
        descent_mode: "gradient" or "adam"
        log: TODO
    Output:
        W, b, w, mu: Weights and activations (offsets) for the ResNet
    '''
    # Padding the input data
    d0, I = np.shape(Y)
    if d0 != d:
        raise Exception("Input must be padded")

    # Initialization

    if theta is None:
        W = np.random.uniform(size=(d, d, K))
        b = np.random.uniform(size=(d, K))
        w = np.random.uniform(size=(d, 1))
        mu = np.random.uniform(size=(1, 1))
    else:
        (W, b, w, mu) = theta

    # W = np.ones((d, d, K)) * 0.5
    # b = np.ones((d, K)) * 0.5
    # w = np.ones((d, 1)) * 0.5
    # mu = np.ones((1, 1)) * 0.5

    if descent_mode == "gradient":
        if tau is None:
            raise Exception("Must specify tau when using gradient descent!")
        descent_state = tau
        descent_function = updateTheta
    elif descent_mode == "adam":
        dim = np.sum([np.prod(W.shape), np.prod(b.shape), np.prod(w.shape), 1])
        descent_state = setupAdam(dim)
        descent_function = adamTheta
    else:
        raise Exception("Illegal descent mode.")

    # Iteration

    it = 0
    J = np.inf
    Js = []

    t = time()

    if isinstance(log, str):
        with open(log, "w") as file:
            file.write("Starting log:\n")

    # best_theta = None
    # best_J = None

    while it < it_max and J > tol:

        # Computing Z's in the forward sweep
        Zs = getZ(Y, K, h, W, b)

        # Applying hypothesis function to recover approximate answer
        Ups = getUpsilon(Zs[:, :, -1], w, mu)

        # Getting diff. in exact and approx and evaluating objective func
        Yc = (Ups.T - c).T
        J = .5 * np.linalg.norm(Yc) ** 2
        Js.append(J)

        # if best_J is None or J < best_J:
        #     best_theta = (W, b, w, mu)
        #     best_J = J

        # Preparing for back-propagation, getting PK = dJ/dZK
        PK = getPK(Yc, Zs[:, :, -1], w, mu)
        # print("PK: {}".format(np.shape(PK)))

        # Getting all Pk for back-propagation
        Ps = getP(PK, Zs, h, W, b)
        # Computing gradient of obj.func. for use in update
        dJ = getdelJ(Ups, c, Ps, Zs, h, W, b, w, mu)

        # Updating the weights and activations in the model
        # Update-scheme followinmg from details in the project
        # W, b, w, mu = updateTheta(.2, dJ, W, b, w, mu)
        (descent_state, W, b, w, mu) = descent_function(
            descent_state, dJ, W, b, w, mu)

        # Another iteration complete!
        it += 1

        if (log) and (time() - t) > 10:
            t += 10
            message = f"{it} / {it_max}: {it / it_max * 100:.1f}%, \
                        order of error: {np.log10(J):.3f}"
            if isinstance(log, str):
                with open(log, "a") as file:
                    file.write(message)
                    file.write("\n")
            else:
                print(message)

    # return (*best_theta, Js)
    return (W, b, w, mu, Js)


# TODO: Make better name or something


def make_scaled_modfunc_and_grad(
        theta,
        y_min,
        y_max,
        c_min,
        c_max,
        h=1.0,
        padding_mode="zeros"):
    alpha, beta = 0.2, 0.8
    (W, b, w, mu) = theta
    modfunc = make_model_function(
        W.shape[2], h, W, b, w, mu, make_padding_function(
            padding_mode, W.shape[0]))

    def scaled_modfunc(Y):
        '''Function representation of the ANN as function approximator

        Input:
            Y: d0xI matrix of input data to be evaluated
        Output:
            ups: output values approximate to exact values
        '''
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
        Y_scaled = 1 / (y_max - y_min) * ((y_max - Y)
                                          * alpha + (Y - y_min) * beta)

        # Compute the scaled output
        ups_scaled = modfunc(Y_scaled)

        # Apply the inverse scaling to the output
        ups = 1 / (beta - alpha) * ((beta - ups_scaled) * c_min
                                    + (ups_scaled - alpha) * c_max)

        return ups

    # Gradient
    def numerical_modfunc(Y):
        '''Function representation of the numerical gradient of the ANN

        Input:
            Y: d0xI matrix of input data where gradient is to be evaluated
        Output:
            nabla_y: gradient of the scalar function which is the ANN
        '''
        dy = 1e-6
        return np.array([((scaled_modfunc((y
                                           + np.identity(len(y))
                                           * dy
                                           / 2).T)
                           - scaled_modfunc((y
                                             - np.identity(len(y))
                                             * dy
                                             / 2).T))
                          / dy) for y in Y.T]).T

    return scaled_modfunc, numerical_modfunc


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
        log=False,
        theta=None,
        y_min=None,
        y_max=None,
        c_min=None,
        c_max=None):
    '''Performs training of a ResNet ANN and wraps functions around both
    the function approximator and gradient for ease of use.

    Input:
        Y: d0xI matrix of I unscaled sets of input
        c: I-vector of exact values for the corresponding input sets
        d: dimension of hidden layers
        K: number of hidden layers
        h: stepsize between layers
        it_max: maximal number of iterations before forcefully stopping
        tol: target tolerance for error of the ANN
        tau: learning factor, must be specified if descent mode is gradient
        descent_mode: "gradient" or "adam
        padding_mode: "zeros", "tiling" or "repeat", see make_padding_function
    Output:
        scaled_modfunc: function representation of the ANN
        gradient_modfunc: gradient of function approximator
        Js: errors from each iteration (useful for gauging effectiveness)
    '''
    # TODO:
    # - make everything be able to take the activation and hypothesis
    # functions as parameters

    # Values to scale between
    alpha, beta = 0.2, 0.8

    # Save the min and max y values for scaling both the training data
    # and the input of the resulting model function
    if y_min is None:
        y_min, y_max = np.min(Y), np.max(Y)

    # Scale the training data input
    Y = 1 / (y_max - y_min) * ((y_max - Y) * alpha + (Y - y_min) * beta)

    # Save the min and max training output data for scaling the training data
    # and inverse scaling the output of the model function
    if c_min is None:
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

    (W, b, w, mu, Js) = trainANN(d, K, h, Y, c, it_max, tol,
                                 tau=tau, descent_mode=descent_mode, log=log,
                                 theta=theta)

    # modfunc = make_model_function(K, h, W, b, w, mu, pad_func)

    scaled_modfunc, numerical_modfunc = make_scaled_modfunc_and_grad(
        (W, b, w, mu), y_min, y_max, c_min, c_max, h=h,
        padding_mode=padding_mode)

    # def scaled_modfunc(Y):
    #     # The padding functions do not like getting one dimensional input
    #     # but we want to be able to use our model function as a function
    #     # of a single vector, or a whole matrix of vector, so in case it's
    #     # a vector, it is simply reshaped to a matrix with a single column
    #     # or in the very special case its a single number, it is reshaped
    #     # to a 1x1 matrix
    #     if isinstance(Y, float):
    #         Y = np.reshape(Y, (1, 1))
    #     elif len(Y.shape) == 1:
    #         Y = np.reshape(Y, (len(Y), 1))

    #     # Scale the input
    #     Y_scaled = 1 / (y_max - y_min) * ((y_max - Y)
    #                                       * alpha + (Y - y_min) * beta)

    #     # Compute the scaled output
    #     c_scaled = modfunc(Y_scaled)

    #     # Apply the inverse scaling to the output
    #     c = 1 / (beta - alpha) * ((beta - c_scaled)
    #                               * c_min + (c_scaled - alpha) * c_max)

    #     return c

    def gradient_modfunc(Y):
        '''Function representation of the gradient of the ANN

        Input:
            Y: d0xI matrix of input where the gradient is to be evaluated
        Output:
            dups: dxI matrix where each column is the gradient of the ANN
                  in the corresponding input value
        '''
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

        Y_scaled = 1 / (y_max - y_min) * ((y_max - Y)
                                          * alpha + (Y - y_min) * beta)

        # gradent of the scaled c with respect to the scaled input Y
        dups_scaled_scaled = getGradANN(Y_scaled, h, W, b, w, mu)

        # gradient of the scaled c with respect to the unscaled input Y
        # scale_factor = (c_max - c_min) / (y_max - y_min)
        dups = dups_scaled_scaled  # * scale_factor
        # print(f"Scaling with: {scale_factor:.5f}")

        return dups

    # def numerical_modfunc(Y):
    #     dy = 1e-6
    #     return np.array([((scaled_modfunc((y +
    #                                        np.identity(len(y)) *
    #                                        dy /
    #                                        2).T) -
    #                        scaled_modfunc((y -
    #                                        np.identity(len(y)) *
    #                                        dy /
    #                                        2).T)) /
    #                       dy) for y in Y.T]).T

    # Return the scaled model function and the evolution of the
    # objective function for analisys
    return (scaled_modfunc, numerical_modfunc, Js, (W, b, w, mu))
