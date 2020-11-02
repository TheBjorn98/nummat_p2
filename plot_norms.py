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
    bNum = 47
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
    l = 1
    off = 0
    dt = 20 / (l * k - off)
    its = l * k - off
    p0 = np.reshape(Ps[:, off], (3, 1))
    q0 = np.reshape(Qs[:, off], (3, 1))

    bEuler = True
    bStrVer = True

    t6 = time.time()
    eulPs, eulQs = im.intMeth(p0, q0, dT, dV, its, euler, dt)
    eulerV = V(eulQs)
    eulerT = T(eulPs)
    eulerH = eulerV + eulerT
    t7 = time.time()
    print(f"Integrated with Euler in {t7 - t6:.3f} s")
    eulQNorm = np.array([np.linalg.norm(y) for y in eulQs.T])
    eulPNorm = np.array([np.linalg.norm(y) for y in eulPs.T])

    t7 = time.time()
    strPs, strQs = im.intMeth(p0, q0, dT, dV, its, strVer, dt)
    strVerV = V(strQs)
    strVerT = T(strPs)
    strH = strVerV + strVerT
    t8 = time.time()
    print(f"Integrated with Størmer-Verlet in {t8 - t7:.3f} s")
    strQNorm = np.array([np.linalg.norm(y) for y in strQs.T])
    strPNorm = np.array([np.linalg.norm(y) for y in strPs.T])

    trueQNorm = np.array([np.linalg.norm(y) for y in Qs.T])
    truePNorm = np.array([np.linalg.norm(y) for y in Ps.T])


    t0, tf = np.min(data["t"]), np.max(data["t"])
    trueTime = np.linspace(t0, tf, k)
    intTime = np.linspace(t0, tf, l * k + 1)[off:]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(trueTime, trueQNorm, label="Norm of Q", linestyle="--")
    ax1.plot(intTime, eulQNorm, label="Norm of Euler's Q")
    ax1.plot(intTime, strQNorm, label="Norm of Størmer-Verlet's Q")
    
    ax2.plot(trueTime, truePNorm, label="Norm of P", linestyle="--")
    ax2.plot(intTime, eulPNorm, label="Norm of Euler's P")
    ax2.plot(intTime, strPNorm, label="Norm of Størmer-Verlet's P")

    ax1.set_xlim(t0, tf)
    ax2.set_xlim(t0, tf)

    ax1.legend(loc="lower right", prop={"size": 6})
    ax2.legend(loc="lower right", prop={"size": 6})

    bSave = False

    ax1.set_ylabel("Position, q")
    # ax1.set_title(f"Norms of p and q for batch {bNum}")
    ax2.set_ylabel("Momentum, p")
    ax2.set_xlabel("Time, t")

    if bSave:
        plt.savefig(f"traj_plots/pq_norm_batch_{bNum}.pdf")
        plt.savefig(f"traj_plots/pq_norm_batch_{bNum}.png")
    plt.show()
