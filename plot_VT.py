import numpy as np
import ann
import matplotlib.pyplot as plt
import pickle
import get_traj_data as gtd

# all_data = get_traj_data.concatenate(0, 50)

(qmin,qmax, vmin, vmax,pmin,pmax, tmin, tmax) = gtd.get_data_bounds()
dStart, dEnd = 40, 50
# dStart, dEnd = 0, 10
# data = gtd.generate_data(2)
data = gtd.concatenate(dStart, dEnd)

with open("thetas/theta_T_new.pickle", "rb") as file:
    theta_T = pickle.load(file)

with open("thetas/theta_V_new.pickle", "rb") as file:
    theta_V = pickle.load(file)

(T, dT) = ann.make_scaled_modfunc_and_grad(theta_T, pmin, pmax, tmin, tmax)
(V, dV) = ann.make_scaled_modfunc_and_grad(theta_V, qmin, qmax, vmin, vmax)

Qs, Vs = data["Q"], data["V"]
Ps, Ts = data["P"], data["T"]

k = len(Qs.T)
t0, tf = np.min(data["t"]), np.max(data["t"])
trueTime = np.linspace(t0, tf, k)

annV = V(Qs)
annT = T(Ps)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

xcoords = [i * 2048 for i in range(dEnd + 1)]

for i in range(dEnd + 1):
    ax1.axvline(i * 2048, color="k")
    ax2.axvline(i * 2048, color="k")

ax1.plot(Vs, label="True V")
ax1.plot(annV, label="Approx. V")
# ax1.plot(Vs + Ts, color="k", linestyle="--")
ax2.plot(Ts, label="True T")
ax2.plot(annT, label="Approx. T")
# ax2.plot(Vs + Ts, color="k", linestyle="--")

ax1.set_ylabel("Potential, V")
ax2.set_ylabel("Kinetic, T")
ax1.set_xlabel("Time, t")
# ax1.set_title(f"True and approximate T and V for batches {dStart} to \
# {dEnd - 1}")

ax1.set_xlim(0, k)
ax2.set_xlim(0, k)

# plt.legend(loc="best")
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

bSave = True
if bSave:
    plt.savefig(f"traj_plots/VT_batch_{dStart}_{dEnd}.pdf")
    plt.savefig(f"traj_plots/VT_batch_{dStart}_{dEnd}.png")

plt.show()
