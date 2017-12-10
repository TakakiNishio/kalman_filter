import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
sns.set_context('poster')

import time


class Plant:

    def __init__(self, m, k):
        self.m = m
        self.k = k

    def u(self, t):
        #u_ = 4.0*np.abs(signal.sawtooth(t*np.sqrt(2)))+10.0*np.sin(t)
        u_ = 4.0*signal.sawtooth(t*np.sqrt(2))+10.0*np.sin(t)
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
        self.dT = dT
        self.Q = Q
        self.R = R
        self.prev_xh = xh0
        self.prev_P = P0
        self.n = len(xh0)

        self.kappa = 0.0
        self.B = B

    def forward(self, y, k):

        P_ = np.zeros((self.n,self.n))

        ## calculation of sigma points
        x_sigma = []
        x_sigma.append(self.prev_xh)

        n_plus_kappa = self.n + self.kappa
        n_kappa_square_root = np.sqrt(n_plus_kappa)
        prev_P_col =  np.linalg.cholesky(self.prev_P)

        print "aaa"
        print prev_P_col

        for i in range(self.n):
            x_sigma.append(self.prev_xh + (n_kappa_square_root * prev_P_col[:,i]))

        for i in range(self.n):
            x_sigma.append(self.prev_xh - (n_kappa_square_root * prev_P_col[:,i]))

        w0 = self.kappa/n_plus_kappa
        wi = 1.0/(2.0*n_plus_kappa)

        x_sigma_ = []

        ## prediction step
        for i in range(self.n*2):
            xxx = self.f((k-1)*self.dT, x_sigma[i].reshape(self.n))
            x_sigma_.append(xxx)

        xh_ = w0 * x_sigma_[0]
        for i in range(1,self.n*2):
            xh_ += wi * x_sigma_[i]

        xx = (x_sigma_[0]-xh_).reshape(1, self.n)
        xxt = xx[:,np.newaxis].reshape(self.n,1)
        P_ = w0 * np.dot(xxt, xx) + self.Q

        for i in range(1, self.n*2):
            xx = (x_sigma_[i]-xh_).reshape(1, self.n)
            xxt = xx[:,np.newaxis].reshape(self.n,1)
            P_ += wi * np.dot(xxt, xx) + self.Q

        #* np.array(np.dot(self.B.T,self.B))[0][0]

        x_sigma_re = []
        x_sigma_.append(xh_)
        P_col =  np.linalg.cholesky(P_)
        for i in range(self.n):
            x_sigma_re.append(xh_ + n_kappa_square_root * P_col[:,i])

        for i in range(self.n):
            x_sigma_re.append(xh_ - n_kappa_square_root * P_col[:,i])

        y_sigma_ = []
        for i in range(2*self.n):
            y_sigma_.append(self.plant.h(x_sigma_re[i]))

        yh_ = w0 * y_sigma_[0]
        for i in range(1,2*self.n):
            yh_ += wi * y_sigma_[i]

        P_yy = w0 * (y_sigma_[0] - yh_)**2
        P_xy = w0 * (y_sigma_[0] - yh_) * (x_sigma_re[0] - xh_).reshape(1, self.n)

        for i in range(1,2*self.n):
            P_yy += wi * (y_sigma_[i]-yh_)**2
            P_xy += wi * (y_sigma_[i]-yh_) * (x_sigma_re[i]-xh_).reshape(1, self.n)

        G = (1.0/(P_yy + self.R))*P_xy

        ## filtering step
        xh = xh_ + G * (y - yh_)

        self.prev_xh = xh
        self.prev_P = P_ - np.dot(G[:,np.newaxis].reshape(self.n,1), P_xy.reshape(1,self.n)).T

        print "bbbbbbb"
        print self.prev_P

        #time.sleep(0.5)
        print xh.shape
        return xh


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

    plant = Plant(m,k)
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
    yh[0] = plant.h(xh[0])
    P0 = np.diag([10, 10, 10])

    ukf = UKF(plant, f, b, dT, Q, R, xh[0], P0)

    # # # time update of estimation data
    for ki in range(1,N):
    #     x[k] = plant.f(x[k-1]) + plant.b * v[k-1]
    #     y[k] = plant.h(x[k]) + w[k]
        xh[ki] = ukf.forward(y[ki],ki)

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
