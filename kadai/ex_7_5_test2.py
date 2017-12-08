import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
sns.set_context('poster')

import time


class Plant:

    def __init__(self, c, m, k):
        self.c = c
        self.m = m
        self.k = k

    def u(self, t):
        u_ = 4.0*np.abs(signal.sawtooth(t*np.sqrt(2)))+10.0*np.sin(t)
        return u_

    def dxdt(self, t, x):
        dxdt_ = np.array([x[1], (-x[2]*x[1]/self.m)-(self.k*x[0]/self.m), 0.0]) \
                + np.array([0.0, 1.0/self.m, 0.0])*self.u(t)
        return dxdt_

    def h(self, x):
        return x[0]


class RK4:

    def __init__(self, plant):
        self.plant = plant
        self.hv = 0.01

    def __call__(self, t, x):
        k1 = self.plant.dxdt(t,x)
        k2 = self.plant.dxdt(t+(self.hv/2.0), x+(self.hv/2.0)*k1)
        k3 = self.plant.dxdt(t+(self.hv/2.0), x+(self.hv/2.0)*k2)
        k4 = self.plant.dxdt(t+self.hv, x+(self.hv*k3))
        x_new = x + (self.hv/6.0) * (k1+(2.0*k2)+(2.0*k3)+k4)
        return x_new


class UKF:

    def __init__(self, plant, f, B, dT, Q, R, xh0, P0):

        self.plant = plant
        self.f = f
        self.h = plant.h
        self.dT = dT
        self.Q = Q
        self.R = R
        self.prev_xh = xh0
        self.prev_P = P0
        self.n = len(xh0)

        self.kappa = 0.0
        self.B = B

        w0 = self.kappa/(self.n+self.kappa)
        wi = 1/(self.n+self.kappa)

        self.W = np.diag([wi for i in range(self.n)])
        self.W[0][0] = w0


    def ut(self, F, xm, Pxx, k):

        xm_t = xm[:,np.newaxis]
        #xm_t = xm
        L = np.linalg.cholesky(Pxx)

        X1 = xm
        #Xa = np.dot(np.ones((self.n,1)),xm).reshape((self.n,self.n))
        Xa = np.dot(np.ones((self.n,1)),xm_t).reshape((self.n,self.n))
        Xb = np.sqrt(self.n+self.kappa)*L
        X = np.hstack((X1, Xa,Xb))

        Y = np.array([F((k)*self.dT, x) for x in X.T])

        # print self.W.shape
        # print Y.T.shape

        ym = (np.dot(self.W,Y.T)).sum(axis=0)

        print ym.shape
        print xm.shape

        print Y.shape
        print X.shape

        Yd = np.array([y - ym[:,np.newaxis] for y in Y.T])
        Xd = np.array([x - xm[:,np.newaxis] for x in X.T])

        print Yd.shape
        print Xd.shape

        Pyy = np.dot(Yd.T,self.W,Yd)
        Pxy = np.dot(Xd.T,self.W,Xd)

        return ym, Pyy, Pxy


    def forward(self, y, k, xhat, P):

        xhat = xhat[:,np.newaxis]
        # y = y[:,np.newaxis]

        xhatm, Pm = self.ut(self.f, xhat, P, k)

        Pm = Pm + np.dot(self.B,self.Q,self.B.T)

        yhatm, Pyy, Pxy = self.ut(self.h, xhatm, Pm, k)

        G = np.dot(Pxy,np.linalg.inv(Pyy+self.R))

        xhat_new = xhatm + np.dot(G,(y-yhatm))

        P_new = Pm - np.dot(G,Pxy)

        return xhat_new, P_new




if __name__ == '__main__':

    ## Problem setting
    # physical parameters
    b = 1.0
    c = 1.0
    m = 2.0
    k = 0.7

    # discretization period
    dT = 0.01

    # input settings
    N = 2000
    t = dT * np.array(range(N))

    plant = Plant(c,m,k)
    R = 0.01

    f = RK4(plant)

    x = np.zeros((N,3))
    y0 = np.zeros(N)

    x[0] = [0.0, 0.0, c]
    y0[0] = plant.h(x[0])

    for ki in range(1,N):
        x[ki] = f((ki-1)*dT,x[ki-1])
        y0[ki] = plant.h(x[ki])

    w = np.random.normal(size=N)*np.sqrt(R)

    y = y0 + w

    Q = np.diag([1e-5,1e-5,1e-5])

    xh = np.zeros((N,3))
    yh = np.zeros(N)

    xh[0] = [0.0, 0.0, 0.1*c]
    yh[0] = plant.h(x[0])
    P = np.diag([10, 10, 10])

    ukf = UKF(plant, f, b, dT, Q, R, xh[0], P)

    # # # time update of estimation data
    for ki in range(1,N):
    #     x[k] = plant.f(x[k-1]) + plant.b * v[k-1]
    #     y[k] = plant.h(x[k]) + w[k]
        xh[ki],P = ukf.forward(y[ki],ki,xh[ki-1],P)

    print xh.shape

    # graph plot
    fig = plt.figure(figsize=(16,9))
    ax1 = plt.subplot2grid((3,1), (0,0))
    ax2 = plt.subplot2grid((3,1), (1,0))
    ax3 = plt.subplot2grid((3,1), (2,0))

    lines1, = ax1.plot(t,x[:,0], 'r', label='True Value')
    # lines1, = ax1.plot(t,y, 'r', label='True Value')
    lines2, = ax1.plot(t,xh[:,0], 'b', label='Estimated Value')
    #ax1.set_ylim((-15, 15))
    ax1.set_xlabel(r'$t$ [s]', fontname='serif', fontsize=23)
    ax1.set_ylabel(r'position', fontname='serif', fontsize=23)
    ax1.legend(loc=1)

    lines3, = ax2.plot(t,x[:,1], 'r', label='True Value')
    lines4, = ax2.plot(t,xh[:,1], 'b', label='Estimated Value')
    #ax2.set_ylim((-10, 10))
    ax2.set_xlabel(r'$t$'+'[s]', fontname='serif', fontsize=23)
    ax2.set_ylabel(r'velocity', fontname='serif', fontsize=23)
    ax2.legend(loc=1)

    lines5, = ax3.plot(t,x[:,2], 'r', label='True Value')
    lines6, = ax3.plot(t,xh[:,2], 'b', label='Estimated Value')
    #ax3.set_ylim((-5, 5))
    ax3.set_xlabel(r'$t$'+'[s]', fontname='serif', fontsize=23)
    ax3.set_ylabel(r'parameter', fontname='serif', fontsize=23)
    ax3.legend(loc=1)
    plt.tight_layout()

    plt.show()
