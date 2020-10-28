import os
import subprocess

import numpy as np
import itertools

import time

import pickle

def make_idun_train_ann_test_job(dim, index, I, d, K, h, tau, it_max, tol, name):
    filetext = f"""#!/bin/sh

#SBATCH --partition=CPUQ
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000MB
#SBATCH --job-name="{name}_Nummat_Neural_training"
#SBATCH --output=out.txt

module purge
module load GCCcore/.9.3.0
module load Python/3.8.2

python3 ../../anntrainingtest.py {dim} {index} {I} {d} {K} {h} {tau} {it_max} {tol} {name}
"""
    if not os.path.exists(f"{name}/"):
        subprocess.call(f"mkdir {name}", shell=True)
    else:
        subprocess.call(f"rm {name}/*", shell=True)
    with open(f"{name}/run.sh", "w") as f:
        f.write(filetext)


def do_1d_tests():
    i = range(0, 3)
    I = range(5, 16, 5)
    d = range(2, 7, 2)
    K = range(4, 9, 2)
    h = [1]
    # tau = np.logspace(-3, 0, 4)
    tau = [0.01, 0.05, 0.1]
    it_max = 1000000
    tol = 1e-5
    
    os.chdir("idun_files")
    
    iterator = itertools.product(i, I, d, K, h, tau)
    
    lookup_table = {}
    
    it_ind = 0
    for x in itertools.product(i, I, d, K, h, tau):
        it_ind += 1
        lookup_table[x] = it_ind
    
    with open("table.pickle", "wb") as pickle_file:
        pickle.dump(lookup_table, pickle_file)
    
    print(f"Preping for {len(list(iterator))} amount of jobs")
    time.sleep(1)
    
    it_ind = 0
    for (i, I, d, K, h, tau) in itertools.product(i, I, d, K, h, tau):
        it_ind += 1
        name = f"Job_{it_ind}"
        make_idun_train_ann_test_job(1, i, I, d, K, h, tau, it_max, tol, name)
        os.chdir(name)
        subprocess.call("sbatch run.sh", shell=True)
        os.chdir("..")
        # if it_ind > 3:
        #     break


if __name__ == "__main__":
    do_1d_tests()
