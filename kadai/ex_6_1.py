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

    # time-series data
    y[0] = c.transpose()*x[0] + w[0]

    for k in range(1,N):
        x[k] = A*x[k-1] + b*v[k-1]
        y[k] = c.transpose()*x[k] + w[k]

    ## estimation using Kalman filter
    # initialization
    P0 = 0.0
    xh[0] = 0.0
    kf = Kalman(A, b, c, Q, R, xh[0], P0)

    # time update of estimation data
    for k in range(1,N):
        xh[k] = kf.forward(y[k])

    # graph plot
    plt.plot(y, 'y', label='Observed Value')
    plt.plot(x, 'r--', label='True Value')
    plt.plot(xh, 'b--', label='Estimated Value')
    plt.xlabel(r'$k$', fontname='serif', fontsize=25)
    plt.legend(loc='best')
    plt.tight_layout()

plt.show()
