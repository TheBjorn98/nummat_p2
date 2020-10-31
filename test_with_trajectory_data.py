import numpy as np
import ann
import matplotlib.pyplot as plt
import pickle
import get_traj_data

all_data = get_traj_data.concatenate(0, 50)

Q_min, Q_max = np.min(all_data["Q"]), np.max(all_data["Q"])
V_min, V_max = np.min(all_data["V"]), np.max(all_data["V"])

P_min, P_max = np.min(all_data["P"]), np.max(all_data["P"])
T_min, T_max = np.min(all_data["T"]), np.max(all_data["T"])

with open("test_pickles/theta_T.pickle", "rb") as file:
    theta_T = pickle.load(file)

with open("test_pickles/theta_V.pickle", "rb") as file:
    theta_V = pickle.load(file)

(T, dT) = ann.make_scaled_modfunc_and_grad(theta_T, P_min, P_max, T_min, T_max)
(V, dV) = ann.make_scaled_modfunc_and_grad(theta_V, Q_min, Q_max, V_min, V_max)

Qs, Vs = all_data["Q"], all_data["V"]
Ps, Ts = all_data["P"], all_data["T"]

k = 10000

V_ann = V(Qs)[0:k]
T_ann = T(Ps)[0:k]
H_ann = V_ann + T_ann
Hs = (Vs + Ts)[0:k]

plt.plot(H_ann)
plt.plot(Hs)
plt.show()
