import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')


class Kalman:

    def __init__(self, A, B, C, Q, R, xh0, P0):

        self.A = np.matrix(A)
        self.B = np.matrix(B)
        self.C = np.matrix(C)
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)
        self.prev_xh = xh0
        self.prev_P = P0

    def forward(self, y):

        ## prediction step
        xh_ = self.A * self.prev_xh
        P_ = self.A * self.prev_P * self.A.T + self.B * self.Q * self.B.T

        ## filtering step
        G = (P_ * self.C.T) * ((self.C * P_ * self.C.T + self.R).I)
        xh = xh_ + G * (y - self.C * self.prev_xh)

        self.prev_xh = xh
        self.prev_P = (np.matrix(np.identity(len(self.A)))-G*self.C)*P_

        return xh


if __name__ == '__main__':

    ## problem setting
    A = np.array([1.0])
    b = np.array([1.0])
    c = np.array([1.0])
    Q = 1.0
    R = 2.0
    N = 300

    ## gereration of observation data
    # noise signal
    v = np.random.normal(size=N)*np.sqrt(Q)
    w = np.random.normal(size=N)*np.sqrt(R)

    x = np.zeros(N)
    y = np.zeros(N)
    xh = np.zeros(N)
    k_step = range(0,N)

    # time-series data
    y[0] = c.transpose()*x[0] + w[0]

    ## estimation using Kalman filter
    # initialization
    P0 = 0.0
    xh[0] = 0.0
    kf = Kalman(A, b, c, Q, R, xh[0], P0)

    # time update of estimation data
    for k in range(1,N):
        x[k] = A*x[k-1] + b*v[k-1]
        y[k] = c.transpose()*x[k] + w[k]
        xh[k] = kf.forward(y[k])

    # graph plot
    fig, ax = plt.subplots(1, 1)
    lines1, = ax.plot(k_step,y, 'y', label='Observed Value')
    lines2, = ax.plot(k_step,x, 'r', label='True Value')
    lines3, = ax.plot(k_step,xh, 'b', label='Estimated Value')
    plt.xlabel(r'$k$', fontname='serif', fontsize=25)
    plt.legend(loc=1)
    plt.tight_layout()

    ## update
    while True:

        prev_step = k_step[-1]
        prev_x = x[-1]

        k_step = np.roll(k_step, -1)
        x = np.roll(x, -1)
        y = np.roll(y, -1)
        xh = np.roll(xh, -1)

        v = np.random.normal()*np.sqrt(Q)
        w = np.random.normal()*np.sqrt(R)

        k_step[-1] = prev_step + 1
        x[-1] = A*prev_x + b*v
        y[-1] = c.transpose()*x[-1] + w
        xh[-1] = kf.forward(y[-1])

        lines1.set_data(k_step,y)
        lines2.set_data(k_step,x)
        lines3.set_data(k_step,xh)
        ax.set_xlim((k_step.min(), k_step.max()))
        ax.set_ylim((xh.min()-5, xh.max()+5))

        plt.pause(0.01)
