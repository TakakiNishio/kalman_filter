import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
sns.set_context('poster')


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
        self.h = 0.01

    def __call__(self, t, x):
        k1 = self.plant.dxdt(t,x)
        k2 = self.plant.dxdt(t+(self.h/2), x+(self.h/2)*k1)
        k3 = self.plant.dxdt(t+(self.h/2), x+(self.h/2)*k2)
        k4 = self.plant.dxdt(t+self.h, x+self.h*k3)
        x_new = x + (self.h/6.0) * (k1+2*k2+2*k3+k4)
        return x_new


class UKF:

    def __init__(self, plant, Q, R, xh0, P0):

        self.plant = plant
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)
        self.prev_xh = xh0
        self.prev_P = P0

        print self.plant.b

    def forward(self, y):

        ## prediction step
        xh_ = self.plant.f(self.prev_xh)
        prev_a = self.plant.a(self.prev_xh)
        c = self.plant.c(xh_)

        P_ = (prev_a**2) * self.prev_P + self.Q * (self.plant.b**2)

        ## filtering step
        G = (P_ * c) / ((c**2) * P_ + self.R)
        xh = xh_ + G * (y - self.plant.h(xh_))

        self.prev_xh = xh
        self.prev_P = (1 - G*c)*P_

        return xh


class EKF:

    def __init__(self, plant, Q, R, xh0, P0):

        self.plant = plant
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)
        self.prev_xh = xh0
        self.prev_P = P0

        print self.plant.b

    def forward(self, y):

        ## prediction step
        xh_ = self.plant.f(self.prev_xh)
        prev_a = self.plant.a(self.prev_xh)
        c = self.plant.c(xh_)

        P_ = (prev_a**2) * self.prev_P + self.Q * (self.plant.b**2)

        ## filtering step
        G = (P_ * c) / ((c**2) * P_ + self.R)
        xh = xh_ + G * (y - self.plant.h(xh_))

        self.prev_xh = xh
        self.prev_P = (1 - G*c)*P_

        return xh


if __name__ == '__main__':

    ## Problem setting
    # physical parameters
    c = 1.0
    m = 2.0
    k = 0.7

    # discretization period
    dT = 0.01

    # input settings
    N = 2000
    t = dT * np.array(range(N))

    plant = Plant(c,m,k)
    R = 0.1

    f = RK4(plant)

    x = np.zeros((N,3))
    y0 = np.zeros(N)

    x[0] = [0, 0, c]
    y01 = plant.h(x[0])

    for k in range(1,N):
        x[k] = f((k-1)*dT,x[k-1])

    # ## gereration of observation data
    # # noise signal
    # v = np.random.normal(size=N)*np.sqrt(Q)
    w = np.random.normal(size=N)*np.sqrt(R)

    y = y0 + w

    Q = np.diag([1e-5,1e-5,1e-5])

    xh = np.zeros((N,3))
    yh = np.zeros(N)

    xh[0] = [0.0, 0.0, 0.1*c]
    yh[0] = plant.h(x[0])
    P = np.diag([10,10,10])

    # #initial value
    # x[0] = 10
    # y[0] = plant.h(x[1])

    # ## estimation using Kalman filter
    # # initialization
    # P0 = 0.0
    # xh[0] = x[0] + 1

    # ekf = EKF(plant, Q, R, xh[0], P0)

    # # # time update of estimation data
    # for k in range(1,N):
    #     x[k] = plant.f(x[k-1]) + plant.b * v[k-1]
    #     y[k] = plant.h(x[k]) + w[k]
    #     xh[k] = ekf.forward(y[k])

    # graph plot
    fig = plt.figure(figsize=(16,9))
    ax1 = plt.subplot2grid((3,1), (0,0))
    ax2 = plt.subplot2grid((3,1), (1,0))
    ax3 = plt.subplot2grid((3,1), (2,0))

    lines1, = ax1.plot(t,x[:,0], 'r', label='True Value')
    ax1.set_xlabel(r'$t$ [s]', fontname='serif', fontsize=23)
    ax1.set_ylabel(r'position', fontname='serif', fontsize=23)
    ax1.legend(loc=1)

    lines2, = ax2.plot(t,x[:,1], 'r', label='True Value')
    ax2.set_xlabel(r'$t$'+'[s]', fontname='serif', fontsize=23)
    ax2.set_ylabel(r'velocity', fontname='serif', fontsize=23)
    ax2.legend(loc=1)

    lines3, = ax3.plot(t,x[:,2], 'r', label='True Value')
    ax3.set_xlabel(r'$t$'+'[s]', fontname='serif', fontsize=23)
    ax3.set_ylabel(r'parameter', fontname='serif', fontsize=23)
    ax3.legend(loc=1)
    plt.tight_layout()

    plt.show()
