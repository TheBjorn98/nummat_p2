import numpy as np

def symEuler(p0,q0,dTdp,dVdq,h): #Burde sende de inn som np.array()
    q = q0 + h*dTdp(p0)
    p = p0 -  h*dVdq(q)
    return p, q

def stVerlet(p0,q0,dTdp,dVdq,dt):
    p_12 = p0 - dt/2*dVdq(q0)
    q = q0 + dt*dTdp(p_12)
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
        diff = (np.linalg.norm(ps[-2] - ps[-1]) + np.linalg.norm(qs[-2] - qs[-1]))/2 #Marcus fiks denne koden, takk
        it += 1

    return np.array(ps), np.array(qs) #er vel kanskje ikke nødvendig hvis de allerede er arrays..


def nonlinPend(q,p):
    def dTdp(p):
        return p
    def dVdq(q):
        #m = ???
        g = 9.81
        #l = ???
        return m*g*l*np.sin(q)
    return dTdp, dVdq

def keplerTwoBody(q,p): 
    def dTdp(p):
        return p
    def dVdq(q):
        gradVec = [q[0]/(q[0]**2 + q[1]**2)**(3/2), q[1]/(q[0]**2 + q[1]**2)**(3/2)]
        #Manually constructs the gradient vector 
        return np.array(vec)





