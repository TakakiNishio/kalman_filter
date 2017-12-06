#!python
#-*-coding:utf-8-*-
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

    time_step = 0.1
    start_y = 0.0

    mean_y = 0.0
    deviation_y = 1.0

    mean_noize = 0.0
    deviation_noize = 5

    position = start_y
    t = np.arange(0, time_step*50, time_step)
    move = np.random.normal(loc=mean_y, scale=deviation_y, size=len(t)-1)
    y = np.insert(move, 0, start_y)
    y = np.cumsum(y)

    noise = np.random.normal(loc=mean_noize, scale=deviation_noize, size=len(y))
    observed_y = y + noise

    start_obs = observed_y[0]

    n_obs = len(observed_y)
    start_pos = start_y
    start_dev = deviation_y
    dev_q = deviation_y
    dev_r = deviation_noize

    kf = Kalman(start_pos, start_dev, dev_q, dev_r)
    y_ = [kf.forward(start_obs)]

    for i in range(1, n_obs):
        y_.append(kf.forward(observed_y[i]))

    y_ = np.array(y_)

    fig, ax = plt.subplots(1, 1)
    lines1, = ax.plot(t, y, 'r--', label='true y')
    lines2, = ax.plot(t, observed_y, 'y', label='observed y')
    lines3, = ax.plot(t, y_, 'blue' ,label='estimated y')
    plt.xlabel('time step')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.tight_layout()

    while True:

        prev_y = y[-1]
        prev_t = t[-1]

        t = np.roll(t, -1)
        y = np.roll(y, -1)
        observed_y = np.roll(observed_y, -1)
        y_ = np.roll(y_, -1)

        t[-1] = prev_t + time_step
        move = np.random.normal(loc=mean_y, scale=deviation_y)
        y[-1] = prev_y + move
        noise = np.random.normal(loc=mean_noize, scale=deviation_noize)
        observed_y[-1] = y[-1] + noise

        y_[-1] = kf.forward(observed_y[-1])

        lines1.set_data(t, y)
        lines2.set_data(t, observed_y)
        lines3.set_data(t, y_)
        ax.set_xlim((t.min(), t.max()))
        ax.set_ylim((observed_y.min()-5, observed_y.max()+5))

        plt.pause(0.01)
