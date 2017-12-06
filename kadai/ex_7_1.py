import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')


class Plant:

    def __init__(self):
        self.b = 1

    def f(self, x):
        return x + 3 * np.cos(x/10)

    def h(self, x):
        return x**3

    def a(self, x):
        return 1 - (3/10) * np.sin(x/10)

    def c(self, x):
        return 3 * (x**2)


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

    ## problem setting
    plant = Plant()
    Q = 1.0
    R = 100.0
    N = 30

    ## gereration of observation data
    # noise signal
    v = np.random.normal(size=N)*np.sqrt(Q)
    w = np.random.normal(size=N)*np.sqrt(R)

    x = np.zeros(N)
    y = np.zeros(N)
    xh = np.zeros(N)
    k_step = range(0,N)

    #initial value
    x[0] = 10
    y[0] = plant.h(x[1])

    ## estimation using Kalman filter
    # initialization
    P0 = 0.0
    xh[0] = x[0] + 1

    ekf = EKF(plant, Q, R, xh[0], P0)

    # # time update of estimation data
    for k in range(1,N):
        x[k] = plant.f(x[k-1]) + plant.b * v[k-1]
        y[k] = plant.h(x[k]) + w[k]
        xh[k] = ekf.forward(y[k])

    # graph plot
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0))

    lines1, = ax1.plot(k_step,y, 'y', label='Observed Value')
    ax1.set_xlabel(r'$k$', fontname='serif', fontsize=25)
    ax1.set_ylabel(r'$y$', fontname='serif', fontsize=25)

    lines2, = ax2.plot(k_step,x, 'r', label='True Value')
    lines3, = ax2.plot(k_step,xh, 'b', label='Estimated Value')
    ax2.set_xlabel(r'$k$', fontname='serif', fontsize=25)
    ax2.set_ylabel(r'$x$', fontname='serif', fontsize=25)
    ax2.legend(loc=1)
    plt.tight_layout()

    ## update
    while True:

        prev_step = k_step[-1]
        prev_x = x[-1]
        prev_v = v[-1]

        k_step = np.roll(k_step, -1)
        x = np.roll(x, -1)
        y = np.roll(y, -1)

        v = np.roll(v, -1)
        w = np.roll(w, -1)

        xh = np.roll(xh, -1)

        k_step[-1] = prev_step + 1
        v[-1] = np.random.normal()*np.sqrt(Q)
        w[-1] = np.random.normal()*np.sqrt(R)

        x[-1] = plant.f(prev_x) + plant.b * prev_v
        y[-1] = plant.h(x[-1]) + w[-1]
        xh[-1] = ekf.forward(y[-1])

        lines1.set_data(k_step,y)
        lines2.set_data(k_step,x)
        lines3.set_data(k_step,xh)
        ax1.set_xlim((k_step.min(), k_step.max()))
        ax1.set_ylim((y.min()-100, y.max()+100))
        ax2.set_xlim((k_step.min(), k_step.max()))
        ax2.set_ylim((xh.min()-1, xh.max()+1))

        plt.pause(0.1)
