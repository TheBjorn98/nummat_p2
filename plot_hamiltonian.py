import numpy as np
import get_traj_data as gtd
import ann
import pickle
import matplotlib.pyplot as plt
import time
import intmethods as im


if __name__ == "__main__":

    t0 = time.time()
    print("test script started...")

    # data = get_traj_data.concatenate(2, 3)
    bNum = 2
    data = gtd.generate_data(bNum)

    (qmin, qmax, vmin, vmax, pmin, pmax, tmin, tmax) = \
        gtd.get_data_bounds()

    t1 = time.time()
    tLoadData = t1 - t0
    print(f"Loaded data in {tLoadData:.3f} s")

    with open("test_pickles/theta_V.pickle", "rb") as file:
        th_V = pickle.load(file)

    with open("test_pickles/theta_T.pickle", "rb") as file:
        th_T = pickle.load(file)

    t2 = time.time()
    tLoadPickle = t2 - t1
    print(f"Loaded theta for ANN in {tLoadPickle:.3f} s")

    (T, dT) = ann.make_scaled_modfunc_and_grad(th_T, pmin, pmax, tmin, tmax)
    (V, dV) = ann.make_scaled_modfunc_and_grad(th_V, qmin, qmax, vmin, vmax)

    t3 = time.time()
    tGetFunction = t3 - t2
    print(f"Constructed functions from theta in {tGetFunction:.3f} s")

    trueV = data["V"]
    trueT = data["T"]
    trueH = trueV + trueT

    Qs = data["Q"]
    Ps = data["P"]

    t4 = time.time()

    annV = V(Qs)
    annT = T(Ps)
    annH = annV + annT

    t5 = time.time()
    tCompVT = t5 - t4
    print(f"Computed V and T in {tCompVT:.3f} s")

    euler = im.symEuler
    strVer = im.stVerlet

    k = len(Qs.T)
    l = 2
    off = 0
    dt = 20 / (l * k - off)
    its = l * k - off
    p0 = np.reshape(Ps[:, off], (3, 1))
    q0 = np.reshape(Qs[:, off], (3, 1))

    bEuler = True
    bStrVer = True

    if bEuler:
        t6 = time.time()
        eulPs, eulQs = im.intMeth(p0, q0, dT, dV, its, 1e-10, euler, dt)
        eulerV = V(eulQs)
        eulerT = T(eulPs)
        eulerH = eulerV + eulerT
        t7 = time.time()
        print(f"Integrated with Euler in {t7 - t6:.3f} s")
        eulQNorm = np.array([np.linalg.norm(y) for y in eulQs.T])

    if bStrVer:
        t7 = time.time()
        strPs, strQs = im.intMeth(p0, q0, dT, dV, its, 1e-10, strVer, dt)
        strVerV = V(strQs)
        strVerT = T(strPs)
        strH = strVerV + strVerT
        t8 = time.time()
        print(f"Integrated with Størmer-Verlet in {t8 - t7:.3f} s")
        strQNorm = np.array([np.linalg.norm(y) for y in strQs.T])

    trueQNorm = np.array([np.linalg.norm(y) for y in Qs.T])

    t0, tf = np.min(data["t"]), np.max(data["t"])
    trueTime = np.linspace(t0, tf, k)
    intTime = np.linspace(t0, tf, l * k + 1)[off:]

    plt.plot(trueTime, trueH[:k], label="True Hamiltonian")
    plt.plot(trueTime, annH[:k], label="Hamiltonian from ANN")
    if bEuler:
        plt.plot(intTime, eulerH, label="Euler Hamiltonian")
    if bStrVer:
        plt.plot(intTime, strH, label="Størmer-Verlet Hamiltonian")
    plt.legend(loc="lower right", prop={"size": 10})

    # plt.title(f"Hamiltonian time evolution for batch {bNum}")
    plt.xlabel("Time, t")
    plt.ylabel("Energy of H")

    bSave = False
    if bSave:
        plt.savefig(f"traj_plots/hamiltonian_batch_{bNum}.pdf")
        plt.savefig(f"traj_plots/hamiltonian_batch_{bNum}.png")

    plt.show()
