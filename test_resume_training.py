import numpy as np
import matplotlib.pyplot as plt
import ann
import pickle


def Fe(y):
    return 1 - np.cos(y)


def dFe(y):
    return np.sin(y)


n = 11
y_rng = np.pi / 1.5
Y1 = np.reshape(np.linspace(-y_rng, .5 * y_rng, n), (1, n))
Y2 = np.reshape(np.linspace(-.5 * y_rng, y_rng, n), (1, n))

y0 = min(np.min(Y1), np.min(Y2))
yf = max(np.max(Y1), np.max(Y2))
c1, c2 = Fe(Y1), Fe(Y2)
c0 = min(np.min(c1), np.min(c2))
cf = max(np.max(c1), np.max(c2))

d, K, h, tau = 4, 4, 1, .05
it_max, tol = 5000, 1e-4

(_, _, Js1, theta1) = ann.train_ANN_and_make_model_function(
    Y1, c1, d, K, h, it_max, tol, tau=tau,
    y_min=y0, y_max=yf, c_min=c0, c_max=cf)

(F1, dF1) = ann.make_scaled_modfunc_and_grad(
    theta1, y0, yf, c0, cf, h=h)

# b is for binary, which is needed when SERIALIZING
with open("test_pickles/test_theta.pickle", "wb") as file:
    pickle.dump(theta1, file)

# b is for binary, which is needed when SERIALIZING
with open("test_pickles/test_theta.pickle", "rb") as file:
    saved_theta = pickle.load(file)

(_, _, Js2, theta2) = ann.train_ANN_and_make_model_function(
    Y2, c2, d, K, h, it_max, tol, tau=tau,
    y_min=y0, y_max=yf, c_min=c0, c_max=cf, theta=saved_theta)

(F2, dF2) = ann.make_scaled_modfunc_and_grad(
    theta2, y0, yf, c0, cf, h=h)

t = 101
ys = np.reshape(np.linspace(y0, yf, t), (1, t))
cs, dcs = np.reshape(Fe(ys), (1, t)), dFe(ys)
Fs1, Fs2 = F1(ys), F2(ys)
dFs1, dFs2 = dF1(ys), dF2(ys)

bErr, bFn, bdFn = True, True, True

if bErr:
    plt.plot(np.concatenate((Js1, Js2)))
    plt.yscale("log")
    plt.show()
if bFn:
    plt.plot(ys.T, cs.T)
    plt.plot(ys.T, Fs1.T)
    plt.plot(ys.T, Fs2.T)
    plt.show()
if bdFn:
    plt.plot(ys.T, dcs.T)
    plt.plot(ys.T, dFs1.T)
    plt.plot(ys.T, dFs2.T)
    plt.show()
