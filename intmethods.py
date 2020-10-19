import numpy as np

def symEuler(p0,q0,dTdp,dVdq,h):
    q = q0 + h*dTdp(p0)
    p = p0 -  h*dVdq(q)
    return p, q

def stVerlet(p0,q0,dTdp,dVdq,dt):
    p_12 = p0 - dt/2*dVdq(q0)
    q = q + dt*dTdp(p_12)
    p = p_12 - dt/2*dVdq(q)
    return p, q

def intMeth(p,q,dTdp, dVdq, it_max, tol, func, step):

    ps = []
    qs = []
    ps.append(p)
    qs.append(q)

    diff = 0
    it = 0
    while it < it_max and diff > tol:
        p, q = func(p, q, dTdp, dVdq, step)
        ps.append(p)
        qs.append(q)
        diff = ps[-2] - ps[-1] #?????
        it += 1

    return ps, qs

