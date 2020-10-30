# Steep Wan

# Find minmax vals:

import get_traj_data

all_data = get_traj_data.concatenate(0, 50)

Q_min = np.min(all_data["Q"])
Q_max = np.max(all_data["Q"])

V_min = np.min(all_data["V"])
V_max = np.max(all_data["V"])

P_min = np.min(all_data["P"])
P_max = np.max(all_data["P"])

T_min = np.min(all_data["T"])
T_max = np.max(all_data["T"])


# Stteep tue

# Load pickled theta:

import pickle

with open("results/T_trained_4x4/theta.pickle", "rb") as file:
    theta = pickle.load(file)


# stiep thtrie

# meek function

(F, dF) = make_scaled_modfunc_and_grad(theta, P_min, P_max, T_min, T_max)

# streeep frour

# Prefit
