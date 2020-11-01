import csv
import numpy as np
# from ast import literal_eval
# import re

"""
Both of the following functions import data. The output of both functions are
a dictionary containing 5 arrays
    t: the array of av time points
    Q: the position values (q)
    P: the momentum values (p)
    T: the kinetic energy
    V: the potential energy

The data files contain data from 50 different trajectories, i.e. simulation of
the path for a point with some
initial position q0 and momentum p0.

The function generate_data gives you the data from one of these data files,
while the function concatenate
gives you the data from multiple trajectories at once. The default arguments
of concatenate give you all the data
alltogether.

The folder project_2_trajectories must be placed in the same folder as your
program to work. If the folder is in
some other location, the path for this location can be put into the string
start_path.
"""


def generate_data(batch=0):
    '''
    Fetches data from a single .csv file.
    File must be in the `project_2_trajectories_new` folder with the name
    `datalist_batch_i.csv` where i is the appropriate trajectory number.

    Input:
    1. `batch`: the file number, i, to be loaded

    Output:
    1. Dictionary containing the loaded data
        * `t`: time data for the trajectory
        * `Q`: generalized positions for the particle
        * `P`: momentums of the particle
        * `T`: kinetic energies of the particle
        * `V`: potential energies of the particle
    '''
    start_path = ""
    path = start_path + "project_2_trajectories_new/datalist_batch_" + \
        str(batch) + ".csv"
    with open(path, newline="\n") as file:
        reader = csv.reader(file)
        datalist = list(reader)

    N = len(datalist)
    t_data = np.array([float(datalist[i][0]) for i in range(1, N)])
    Q1_data = [float(datalist[i][1]) for i in range(1, N)]
    Q2_data = [float(datalist[i][2]) for i in range(1, N)]
    Q3_data = [float(datalist[i][3]) for i in range(1, N)]
    P1_data = [float(datalist[i][4]) for i in range(1, N)]
    P2_data = [float(datalist[i][5]) for i in range(1, N)]
    P3_data = [float(datalist[i][6]) for i in range(1, N)]
    T_data = np.array([float(datalist[i][7]) for i in range(1, N)])
    V_data = np.array([float(datalist[i][8]) for i in range(1, N)])

    Q_data = np.transpose(
        np.array([[Q1_data[i], Q2_data[i], Q3_data[i]] for i in range(N - 1)]))
    P_data = np.transpose(
        np.array([[P1_data[i], P2_data[i], P3_data[i]] for i in range(N - 1)]))

    return {"t": t_data, "Q": Q_data, "P": P_data, "T": T_data, "V": V_data}


def concatenate(batchmin=0, batchmax=50):
    '''
    Fetches data from a range of .csv files.
    Files must be in the `project_2_trajectories_new` folder with the name
    `datalist_batch_i.csv` where i is the appropriate trajectory numbers.

    Input:
    1. `batchmin`: the lowest filenumber to be loaded
    2. `batchmax`: the highest filenumber, which will not be loaded as the
        range is exclusive of the final number

    Output:
    1. Dictionary containing the loaded data
        * `t`: time data for the trajectory
        * `Q`: generalized positions for the particle
        * `P`: momentums of the particle
        * `T`: kinetic energies of the particle
        * `V`: potential energies of the particle
    '''
    dictlist = []
    for i in range(batchmin, batchmax):
        dictlist.append(generate_data(batch=i))
    Q_data = dictlist[0]["Q"]
    P_data = dictlist[0]["P"]
    T0 = dictlist[0]["T"]
    V0 = dictlist[0]["V"]
    tlist = dictlist[0]["t"]
    for j in range(batchmax - batchmin - 1):
        Q_data = np.hstack((Q_data, dictlist[j + 1]["Q"]))
        P_data = np.hstack((P_data, dictlist[j + 1]["P"]))
        T0 = np.hstack((T0, dictlist[j + 1]["T"]))
        V0 = np.hstack((V0, dictlist[j + 1]["V"]))
        tlist = np.hstack((tlist, dictlist[j + 1]["t"]))
    return {"t": tlist, "Q": Q_data, "P": P_data, "T": T0, "V": V0}


def get_data_bounds():
    '''
    Returns the global min/max values of the datasets found in
    `project_2_trajectories_new` folder. This function relies on the
    `concatenate` function in `get_traj_data.py`.

    Input:
    1. Nothing, this function is absolutely impure

    Output:
    1. `qmin, qmax`: min/max values for q
    2. `vmin, vmax`: min/max values for V
    3. `pmin, pmax`: min/max values for p
    4. `tmin, tmax`: min/max values for T
    '''
    data = concatenate(0, 50)

    qmin, qmax = np.min(data["Q"]), np.max(data["Q"])
    vmin, vmax = np.min(data["V"]), np.max(data["V"])

    pmin, pmax = np.min(data["P"]), np.max(data["P"])
    tmin, tmax = np.min(data["T"]), np.max(data["T"])

    return (qmin, qmax, vmin, vmax, pmin, pmax, tmin, tmax)
