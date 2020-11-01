import numpy as np
import matplotlib.pyplot as plt


def symEuler(p0, q0, dTdp, dVdq, h):  # Burde sende de inn som np.array()
    q = q0 + h * dTdp(p0)
    p = p0 - h * dVdq(q)
    return p, q


def stVerlet(p0, q0, dTdp, dVdq, dt):
    p_12 = p0 - dt / 2 * dVdq(q0)
    q = q0 + dt * dTdp(p_12)
    p = p_12 - dt / 2 * dVdq(q)
    return p, q


def intMeth(p0, q0, dTdp, dVdq, it_max, tol, func, step):
    '''
    Employs a specific integrator method to find the path of a particle
    with the specified hamiltonian given by dTdp and dVdq

    Inputs:
    1. p0: initial momentum of the particle
    2. q0: initial general position of the particle
    3. dTdp: kinetic part of the hamiltonian diffed wrt. momentum
    4. dVdq: potential part of the hamiltonian diffed wrt. position
    5. it_max: number of steps the method will perform
    6. tol: unused parameter, disregard
    7. func: integrator method to be used, Euler or Strømer-Verlet
    8. step: stepsize in time for the integrator method

    Outputs:
    1. ps: momentum as a function of time
    2. qs: position as a function of time
    '''
    ps = []
    qs = []
    ps.append(p0)
    qs.append(q0)
    p, q = p0, q0

    lp, lq = np.shape(p0)[1], np.shape(q0)[1]

    diff = 100
    it = 0
    while it < it_max and diff > tol:
        p, q = func(p, q, dTdp, dVdq, step)
        ps.append(p)
        qs.append(q)
        diff = (np.linalg.norm(ps[-2] - ps[-1])
                + np.linalg.norm(qs[-2] - qs[-1])) / 2
        it += 1

    return (np.reshape(np.array(ps).T, (lp, it_max + 1)),
            np.reshape(np.array(qs).T, (lq, it_max + 1)))


def nonlinPend():
    def dTdp(p):
        return p

    def dVdq(q):
        m = 1
        g = 9.81
        l = 1
        return m * g * l * np.sin(q)

    return dTdp, dVdq


def keplerTwoBody():

    def dTdp(p):
        return p

    def dVdq(q):
        gradVec = [q[0] / (q[0]**2 + q[1]**2)**(3 / 2),
                   q[1] / (q[0]**2 + q[1]**2)**(3 / 2)]
        # Manually constructs the gradient vector
        return np.array(gradVec)
    return dTdp, dVdq


if __name__ == "__main__":
    p0 = np.array([0, 1.3])
    q0 = np.array([1, 0])

    pen_dTdp, pen_dVdq = nonlinPend()
    euler_pen_p, euler_pen_q = intMeth(
        p0[1], q0[1], pen_dTdp, pen_dVdq, 1000, 10**(-4), symEuler, 0.01)
    verlet_pen_p, verlet_pen_q = intMeth(
        p0[1], q0[1], pen_dTdp, pen_dVdq, 1000, 10**(-4), stVerlet, 0.01)

    kep_dTdp, kep_dVdq = keplerTwoBody()
    euler_kep_p, euler_kep_q = intMeth(
        p0, q0, kep_dTdp, kep_dVdq, 10000, 10**(-4), symEuler, 0.01)
    verlet_kep_p, verlet_kep_q = intMeth(
        p0, q0, kep_dTdp, kep_dVdq, 10000, 10**(-4), stVerlet, 0.01)

    plt.plot(euler_pen_q, euler_pen_p)
    plt.xlabel("q")
    plt.ylabel("p")
    plt.title("Plot of the nonlinear pendulum problem with the symplectic \
               Euler method")

    plt.show()

    plt.plot(verlet_pen_q, verlet_pen_p)
    plt.xlabel("q")
    plt.ylabel("p")
    plt.title("Phase plot of the nonlinear pendulum problem with the \
               Stormer-Verlet method")

    plt.show()

    plt.plot(euler_kep_q[:, 0], euler_kep_q[:, 1])
    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.title("The Kepler two-body problem with the symplectic Euler method")

    plt.show()

    plt.plot(verlet_kep_q[:, 0], verlet_kep_q[:, 1])
    plt.xlabel("q1")
    plt.ylabel("q2")
    plt.title("The Kepler two-body problem with the Stormer-Verlet method")

    plt.show()
