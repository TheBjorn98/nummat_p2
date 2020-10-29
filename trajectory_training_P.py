from time import time
import get_traj_data

import ann


import numpy as np
import matplotlib.pyplot as plt

import pickle

import subprocess
import os


all_data = get_traj_data.concatenate(0, 50)

Q_min = np.min(all_data["Q"])
Q_max = np.max(all_data["Q"])

P_min = np.min(all_data["P"])
P_max = np.max(all_data["P"])

V_min = np.min(all_data["V"])
V_max = np.max(all_data["V"])

T_min = np.min(all_data["T"])
T_max = np.max(all_data["T"])


def test_single_batch(Y, c, i, theta=None, tau=0.0001):
    t = time()

    d = 4
    K = 4
    h = 1
    it_max = 5000
    tol = 1e-4

    (F, dF, Js, theta) = ann.train_ANN_and_make_model_function(
        Y, c, d, K, h,
        it_max, tol, tau=tau, padding_mode="zeros", log="log.txt", theta=theta,
        y_min=P_min, y_max=P_max, c_min=T_min, c_max=T_max)

    print(f"took {time() - t:.2f} seconds to train")

    plt.plot(Js)
    plt.yscale("log")
    plt.savefig(f"{i}.png")
    plt.clf()
    plt.cla()

    print(f"J = {Js[-1]}")

    return theta


# foldername = "V_trained_4x4"
# foldername = "V_trained_6x4"
foldername = "T_trained_4x4"


def startup():
    subprocess.call(f"mkdir results/{foldername}", shell=True)
    batch_data = get_traj_data.generate_data(batch=0)
    os.chdir(f"results/{foldername}")
    Y = batch_data["P"][:, :1000]
    c = batch_data["T"][:1000]
    theta = test_single_batch(Y, c, 0, tau=0.001)

    with open("theta.pickle", "wb") as file:
        pickle.dump(theta, file)

    os.chdir("../..")


def resume():
    os.chdir(f"results/{foldername}")
    with open("theta.pickle", "rb") as file:
        theta = pickle.load(file)

    for i in range(0, 21):
        os.chdir("../..")
        batch_data = get_traj_data.generate_data(batch=i)
        os.chdir(f"results/{foldername}")
        Y = batch_data["P"]
        c = batch_data["T"]
        theta = test_single_batch(Y, c, i + 1, theta=theta, tau=0.0001)

    with open("theta.pickle", "wb") as file:
        pickle.dump(theta, file)

    os.chdir("../..")


def resume_with_big_data(lo, hi, plot_index=0):
    os.chdir(f"results/{foldername}")
    with open("theta.pickle", "rb") as file:
        theta = pickle.load(file)

    os.chdir("../..")
    batch_data = get_traj_data.concatenate(lo, hi)
    os.chdir(f"results/{foldername}")

    Y = batch_data["P"]
    c = batch_data["T"]

    theta = test_single_batch(Y, c, plot_index, theta=theta, tau=0.00001)

    with open("theta.pickle", "wb") as file:
        pickle.dump(theta, file)

    os.chdir("../..")


if __name__ == "__main__":
    # startup()
    # resume()
    resume_with_big_data(0, 10, 0)
    resume_with_big_data(5, 15, 1)
    resume_with_big_data(10, 20, 2)
    resume_with_big_data(15, 25, 3)
