# import numpy as np
from intmethods import intMeth, stVerlet, symEuler
import ann
import matplotlib.pyplot as plt
import pickle
import get_traj_data as gtd
import numpy as np

bNum = 2
data = gtd.generate_data(bNum)

(qmin, qmax, vmin, vmax, pmin, pmax, tmin, tmax) = gtd.get_data_bounds()

with open("thetas/theta_T_new.pickle", "rb") as file:
    theta_T = pickle.load(file)

with open("thetas/theta_V_new.pickle", "rb") as file:
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
stPs, stQs = intMeth(p0, q0, dT, dV, its, 1e-10, stVerlet, dt)


pfig, (pax1, pax2, pax3) = plt.subplots(3, 1, sharex=True)

pax1.plot(Ps[0, :k], label="True P1", linestyle="--", color="r")
pax2.plot(Ps[1, :k], label="True P2", linestyle="--", color="g")
pax3.plot(Ps[2, :k], label="True P3", linestyle="--", color="b")
pax1.plot(eulPs[0, :], label="Euler P1", color="r")
pax2.plot(eulPs[1, :], label="Euler P2", color="g")
pax3.plot(eulPs[2, :], label="Euler P3", color="b")
pax1.plot(stPs[0, :], label="Størmer-Verlet P1", color="c")
pax2.plot(stPs[1, :], label="Størmer-Verlet P2", color="m")
pax3.plot(stPs[2, :], label="Størmer-Verlet P3", color="y")


pax1.set_xlim(0, k)
pax2.set_xlim(0, k)
pax3.set_xlim(0, k)
pax1.legend(loc="best")
pax2.legend(loc="best")
pax3.legend(loc="best")
pax1.set_ylabel(r"$p_1$")
pax2.set_ylabel(r"$p_2$")
pax3.set_ylabel(r"$p_3$")

bSave = True
if bSave:
    plt.savefig(f"traj_plots/components_p_batch_{bNum}.pdf")
    plt.savefig(f"traj_plots/components_p_batch_{bNum}.png")

plt.show()

qfig, (qax1, qax2, qax3) = plt.subplots(3, 1)

qax1.plot(Qs[0, :k], label="True Q1", linestyle="--", color="r")
qax2.plot(Qs[1, :k], label="True Q2", linestyle="--", color="g")
qax3.plot(Qs[2, :k], label="True Q3", linestyle="--", color="b")
qax1.plot(eulQs[0, :], label="Euler Q1", color="r")
qax2.plot(eulQs[1, :], label="Euler Q2", color="g")
qax3.plot(eulQs[2, :], label="Euler Q3", color="b")
qax1.plot(stQs[0, :], label="Størmer-Verlet Q1", color="c")
qax2.plot(stQs[1, :], label="Størmer-Verlet Q2", color="m")
qax3.plot(stQs[2, :], label="Størmer-Verlet Q3", color="y")


qax1.set_xlim(0, k)
qax2.set_xlim(0, k)
qax3.set_xlim(0, k)
qax1.legend(loc="best")
qax2.legend(loc="best")
qax3.legend(loc="best")
qax1.set_ylabel(r"$q_1$")
qax2.set_ylabel(r"$q_2$")
qax3.set_ylabel(r"$q_3$")

bSave = True
if bSave:
    plt.savefig(f"traj_plots/components_q_batch_{bNum}.pdf")
    plt.savefig(f"traj_plots/components_q_batch_{bNum}.png")

plt.show()
