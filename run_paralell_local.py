import os
import subprocess

import numpy as np
import itertools

import time

import pickle

import multiprocessing as mp


def make_local_job(
        dim,
        index,
        I,
        d,
        K,
        h,
        tau,
        it_max,
        tol,
        name):
    filetext = f"""
python3 ../../anntrainingtest.py {dim} {index} {I} {d} {K} {h} {tau} {it_max} {tol} {name} > out.txt
"""
    if not os.path.exists(f"{name}/"):
        subprocess.call(f"mkdir {name}", shell=True)
    else:
        subprocess.call(f"rm {name}/*", shell=True)
    with open(f"{name}/run.sh", "w") as f:
        f.write(filetext)


def make_and_run(x):
    (it_ind, (i, I, d, K, h, tau, it_max, tol)) = x
    name = f"Job_{it_ind}"
    make_local_job(1, i, I, d, K, h, tau, it_max, tol, name)
    os.chdir(name)
    subprocess.call("sh run.sh", shell=True)
    os.chdir("..")


def do_1d_tests(idun=False):
    i = [0, 1, 2]
    I = [10, 50]
    d = [2, 4, 8]
    K = [2, 4, 8]
    h = [1]
    tau = [0.01, 0.05, 0.1]
    it_max = [100000]
    tol = [1e-5]

    os.chdir("local_parallell")

    iterator = itertools.product(i, I, d, K, h, tau, it_max, tol)

    l = len(list(iterator))

    print(f"Preping for {l} amount of jobs")
    # time.sleep(1)

    lookup_table = {}

    it_ind = 0
    for x in itertools.product(i, I, d, K, h, tau):
        it_ind += 1
        lookup_table[x] = it_ind

    with open("table.pickle", "wb") as pickle_file:
        pickle.dump(lookup_table, pickle_file)

    with mp.Pool() as pool:
        pool.map(
            make_and_run, list(zip(
                range(l), itertools.product(
                    i, I, d, K, h, tau, it_max, tol))))


if __name__ == "__main__":
    do_1d_tests()
