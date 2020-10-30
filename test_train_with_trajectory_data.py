import numpy as np
import matplotlib.pyplot as plt
import ann
import get_traj_data
import pickle

first_set = get_traj_data.concatenate(0, 1)

Q_min, Q_max = np.min(first_set["Q"]), np.max(first_set["Q"])
V_min, V_max = np.min(first_set["V"]), np.max(first_set["V"])

P_min, P_max = np.min(first_set["P"]), np.max(first_set["P"])
T_min, T_max = np.min(first_set["T"]), np.max(first_set["T"])

d, K, h, tau = 4, 4, 1, .01
it_max, tol = 1000, 1e-4

(V, dV, JsV, theta_V) = ann.train_ANN_and_make_model_function(
    first_set["Q"], first_set["V"], d, K, h, it_max, tol, tau = tau,
    y_min = Q_min, y_max = Q_max, c_min = V_min, c_max = V_max, log=True)

(T, dT, JsT, theta_T) = ann.train_ANN_and_make_model_function(
    first_set["P"], first_set["T"], d, K, h, it_max, tol, tau = tau,
    y_min = P_min, y_max = P_max, c_min = T_min, c_max = T_max, log=True)

# theta_V and theta_T can now be pickled