import numpy as np
import matplotlib.pyplot as plt
import itertools
import ann


def F_exact(y):
    return 1 / 2 * np.linalg.norm(y)**2


def dF_exact(y):
    return np.array([y[0], y[1]]) * np.linalg.norm(y)


n, y_min, y_max = 5, 0, 1
_Y = np.linspace((y_min, y_min), (y_max, y_max), n).T
x, y = np.linspace(0, 1, n), np.linspace(0, 1, n)
Y = np.array(list(itertools.product(x, y))).T
c = np.array([F_exact(Y[:, i]) for i in range(np.shape(Y)[1])])
c_min, c_max = np.min(c), np.max(c)

d, K, h, tau = 4, 4, 1, .1
it_max, tol = 10000, 1e-4

(_, _, Js, theta) = ann.train_ANN_and_make_model_function(
    Y, c, d, K, h, it_max, tol, tau=tau,
    y_min=y_min, y_max=y_max, c_min=c_min, c_max=c_max)

(F, dF) = ann.make_scaled_modfunc_and_grad(
    theta, y_min, y_max, c_min, c_max, h=h)

t = 101
ys = np.linspace((0, 0), (1, 1), t).T
cs = np.array([F_exact(ys[:, i]) for i in range(t)])
dcs = np.array([dF_exact(ys[:, i]) for i in range(t)])
Fs, dFs = F(ys), dF(np.reshape(ys, (2, t)))

bErr, bFn, bdFn = False, False, True

if bErr:
    plt.plot(Js)
    plt.yscale("log")
    plt.show()
if bFn:
    plt.plot(cs)
    plt.plot(Fs)
    plt.show()
if bdFn:
    plt.plot(dcs[:, 0], label="dF, 0-th component")
    plt.plot(dcs[:, 1], label="dF, 1-st component")
    plt.plot(dFs[0, :], label="dF~, 0-th component")
    plt.plot(dFs[1, :], label="dF~, 1-st component")
    plt.legend(loc="best")
    plt.show()
