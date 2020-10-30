import numpy as np
import ann
import matplotlib.pyplot as plt
import get_traj_data

all_data = get_traj_data.concatenate(0, 50)

Q_min = np.min(all_data["Q"])
Q_max = np.max(all_data["Q"])
V_min = np.min(all_data["V"])
V_max = np.max(all_data["V"])

import pickle

with open("test_pickles/theta_V.pickle", "rb") as file:
    theta = pickle.load(file)

(V, dV) = ann.make_scaled_modfunc_and_grad(theta, Q_min, Q_max, V_min, V_max)

Qs, Vs = all_data["Q"], all_data["V"]
V_ann = V(Qs)

k = 10000
plt.plot(V_ann[0:k])  # Plotting the first k values of V from ANN
plt.plot(Vs[0:k])  # Plotting the first k values of V from traj_data
plt.show()