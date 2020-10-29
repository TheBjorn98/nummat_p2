import os
import subprocess

import numpy as np
import itertools

import time

import pickle


def make_idun_train_ann_test_job(
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
    filetext = f"""#!/bin/sh

#SBATCH --partition=CPUQ
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000MB
#SBATCH --job-name="{name}_Nummat_Neural_training"
#SBATCH --output=out.txt

module purge
module load GCCcore/.9.3.0
module load Python/3.8.2

python3 ../../../anntrainingtest.py {dim} {index} {I} {d} {K} {h} {tau} {it_max} {tol} {name}
"""
    if not os.path.exists(f"{name}/"):
        subprocess.call(f"mkdir {name}", shell=True)
    else:
        subprocess.call(f"rm {name}/*", shell=True)
    with open(f"{name}/run.sh", "w") as f:
        f.write(filetext)


def do_tests(dim, i, I, d, K, h, tau, it_max, tol, folder):
    os.chdir("results")
    if os.path.exists(folder):
        subprocess.call(f"rm {folder}/* -r", shell=True)
    else:
        subprocess.call(f"mkdir {folder}", shell=True)
    os.chdir(folder)

    iterator = itertools.product(i, I, d, K, h, tau, it_max, tol)

    lookup_table = {}

    it_ind = 0
    for x in itertools.product(i, I, d, K, h, tau, it_max, tol):
        it_ind += 1
        lookup_table[x] = it_ind

    with open("table.pickle", "wb") as pickle_file:
        pickle.dump(lookup_table, pickle_file)

    print(f"Preping for {len(list(iterator))} amount of jobs")
    time.sleep(1)

    it_ind = 0
    for (
            i,
            I,
            d,
            K,
            h,
            tau,
            it_max,
            tol) in itertools.product(
            i,
            I,
            d,
            K,
            h,
            tau,
            it_max,
            tol):
        it_ind += 1
        name = f"Job_{it_ind}"
        make_idun_train_ann_test_job(
            dim, i, I, d, K, h, tau, it_max, tol, name)
        os.chdir(name)
        subprocess.call("sbatch run.sh", shell=True)
        os.chdir("..")
