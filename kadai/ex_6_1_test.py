import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')


class Kalman:

    def __init__(self, start_position, start_deviation, deviation_position, deviation_noise):

        self.prev_y = start_position
        self.dev_q = deviation_position
        self.dev_r = deviation_noise
        self.prev_P = start_deviation
        self.start_flag = False

    def forward(self, observation):

        if self.start_flag is False:
            self.start_flag = True
        else:
            self.prev_P = self.P_ + self.dev_q

        K = self.prev_P / (self.prev_P + self.dev_r)
        self.P_ = self.dev_r * self.prev_P / (self.prev_P + self.dev_r)
        self.y_ = self.prev_y + K * (observation - self.prev_y)
        self.prev_y = self.y_

        return self.y_


if __name__ == '__main__':

    A = np.array([1.0])
    b = np.array([1.0])
    c = np.array([1.0])
    Q = 1.0
    R = 2.0
    N = 30

    v = np.random.normal(size=N)*np.sqrt(Q)
    w = np.random.normal(size=N)*np.sqrt(R)

    x = np.zeros(N); y = np.zeros(N)
    y[0] = c.transpose()*x[0] + w[0]

    for k in range(1,N):
        x[k] = A*x[k-1] + b*v[k-1]
        y[k] = c.transpose()*x[k] + w[k]

    kf = Kalman(0.0, 0.0, Q, R)
    y_ = [kf.forward(y[0])]

    for i in range(1,N):
        y_.append(kf.forward(y[i]))

    plt.plot(y, 'y', label='Observed Positions')
    plt.plot(x, 'r--', label='True Positions')
    plt.plot(y_, 'b--', label='Estimated Positions')
    plt.title('Random Walk')
    plt.xlabel('time step')
    plt.ylabel('position')
    plt.legend(loc='best')
    plt.tight_layout()

plt.show()
