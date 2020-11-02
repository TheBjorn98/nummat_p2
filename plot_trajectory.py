# import numpy as np
from intmethods import intMeth, symEuler
import ann
import matplotlib.pyplot as plt
import pickle
import get_traj_data as gtd
import numpy as np

data = gtd.generate_data(44)

(qmin, qmax, vmin, vmax, pmin, pmax, tmin, tmax) = gtd.get_data_bounds()

with open("test_pickles/theta_T.pickle", "rb") as file:
    theta_T = pickle.load(file)

with open("test_pickles/theta_V.pickle", "rb") as file:
    theta_V = pickle.load(file)

(T, dT) = ann.make_scaled_modfunc_and_grad(theta_T, pmin, pmax, tmin, tmax)
(V, dV) = ann.make_scaled_modfunc_and_grad(theta_V, qmin, qmax, vmin, vmax)

Qs, Vs = data["Q"], data["V"]
Ps, Ts = data["P"], data["T"]

k = len(Qs.T)
l = 1
m = 1
off = 0

p0 = np.reshape(Ps[:, off], (3, 1))
q0 = np.reshape(Qs[:, off], (3, 1))

its = (k * l - off) // m
dt = 10 / its

eulPs, eulQs = intMeth(p0, q0, dT, dV, its, 1e-10, symEuler, dt)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.plot(Ps[0, :k], label="True P1", linestyle="--", color="r")
ax1.plot(eulPs[0, :], label="Euler P1", color="r")

ax1.plot(Ps[1, :k], label="True P2", linestyle="--", color="g")
ax1.plot(eulPs[1, :], label="Euler P2", color="g")

ax1.plot(Ps[2, :k], label="True P3", linestyle="--", color="b")
ax1.plot(eulPs[2, :], label="Euler P3", color="b")

ax1.set_xlim(0, k)
ax2.set_xlim(0, k)
ax3.set_xlim(0, k)
ax1.legend(loc="best")
ax2.legend(loc="best")
ax3.legend(loc="best")

plt.show()
