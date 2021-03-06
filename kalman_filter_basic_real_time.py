#!python
#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import seaborn as sns
sns.set_context('poster')


class Kalman:

    def __init__(self, observation, start_position, start_deviation, deviation_true, deviation_noise):

        self.obs = observation
        self.n_obs = len(observation)
        self.start_pos = start_position
        self.start_dev = start_deviation
        self.dev_q = deviation_true
        self.dev_r = deviation_noise

        self.x_prev_ = [self.start_pos]
        self.P_prev_ = [self.start_dev]
        self.K_ = [self.P_prev_[0] / (self.P_prev_[0] + self.dev_r)]
        self.P_ = [self.dev_r * self.P_prev_[0] / (self.P_prev_[0] + self.dev_r)]
        self.x_ = [self.x_prev_[0] + self.K_[0] * (self.obs[0] - self.x_prev_[0])]


        self._forward()


    def _forward(self):

        for t in range(1, self.n_obs):

            self.x_prev_.append(self.x_[t-1])
            self.P_prev_.append(self.P_[t-1] + self.dev_q)

            self.K_.append(self.P_prev_[t] / (self.P_prev_[t] + self.dev_r))
            self.P_.append(self.dev_r * self.P_prev_[t] / (self.P_prev_[t] + self.dev_r))
            self.x_.append(self.x_prev_[t] + self.K_[t] * (self.obs[t] - self.x_prev_[t]))


# class Kalman:

#     def __init__(self, observation, start_position, start_deviation, deviation_true, deviation_noise):

#         self.obs = observation
#         self.n_obs = len(observation)
#         self.start_pos = start_position
#         self.start_dev = start_deviation
#         self.dev_q = deviation_true
#         self.dev_r = deviation_noise

#         self._forward()


#     def _forward(self):

#         self.x_prev_ = [self.start_pos]
#         self.P_prev_ = [self.start_dev]
#         self.K_ = [self.P_prev_[0] / (self.P_prev_[0] + self.dev_r)]
#         self.P_ = [self.dev_r * self.P_prev_[0] / (self.P_prev_[0] + self.dev_r)]
#         self.x_ = [self.x_prev_[0] + self.K_[0] * (self.obs[0] - self.x_prev_[0])]

#         for t in range(1, self.n_obs):
#             self.x_prev_.append(self.x_[t-1])
#             self.P_prev_.append(self.P_[t-1] + self.dev_q)

#             self.K_.append(self.P_prev_[t] / (self.P_prev_[t] + self.dev_r))
#             self.P_.append(self.dev_r * self.P_prev_[t] / (self.P_prev_[t] + self.dev_r))
#             self.x_.append(self.x_prev_[t] + self.K_[t] * (self.obs[t] - self.x_prev_[t]))


if __name__ == '__main__':

    time_step = 0.1
    start_y = 0.0

    mean_y = 0.0
    deviation_y = 1.0

    mean_noize = 0.0
    deviation_noize = 5

    position = start_y
    x = np.arange(0, time_step*50, time_step)
    print len(x)
    move = np.random.normal(loc=mean_y, scale=deviation_y, size=len(x)-1)
    y = np.insert(move, 0, start_y)
    y = np.cumsum(y)

    noise = np.random.normal(loc=mean_noize, scale=deviation_noize, size=len(y))
    observed_y = y + noise

    # initialization
    obs = observed_y
    n_obs = len(observed_y)
    start_pos = start_y
    start_dev = deviation_y
    dev_q = deviation_y
    dev_r = deviation_noize

    x_prev_ = [start_pos]
    P_prev_ = [start_dev]
    K_ = [P_prev_[0] / (P_prev_[0] + dev_r)]
    P_ = [dev_r * P_prev_[0] / (P_prev_[0] + dev_r)]
    x_ = [x_prev_[0] + K_[0] * (obs[0] - x_prev_[0])]

    for t in range(1, n_obs):

        x_prev_.append(x_[t-1])
        P_prev_.append(P_[t-1] + dev_q)

        K_.append(P_prev_[t] / (P_prev_[t] + dev_r))
        P_.append(dev_r * P_prev_[t] / (P_prev_[t] + dev_r))
        x_.append(x_prev_[t] + K_[t] * (obs[t] - x_prev_[t]))

    x_prev_ = np.array(x_prev_)
    P_prev_ = np.array(P_prev_)
    K_ = np.array(K_)
    P_ = np.array(P_)
    x_ = np.array(x_)

    fig, ax = plt.subplots(1, 1)
    lines1, = ax.plot(x, y, 'r--', label='true y')
    lines2, = ax.plot(x, observed_y, 'y', label='observed y')
    lines3, = ax.plot(x, x_, 'blue' ,label='estimated y')
    plt.xlabel('time step')
    plt.ylabel('y')
    plt.legend(loc=1)
    plt.tight_layout()


    while True:

        initial_y = y[0]
        prev_x = x[-1]
        prev_y = y[-1]
        prev_observed_y = observed_y[-1]
        prev_P = P_[-1]+dev_q

        x = np.roll(x, -1)
        y = np.roll(y, -1)
        observed_y = np.roll(observed_y, -1)
        P_ = np.roll(P_, -1)
        x_ = np.roll(x_, -1)

        x[-1] = prev_x + time_step
        y[-1] = prev_y + np.random.normal(loc=mean_y, scale=deviation_y)
        observed_y[-1] = y[-1] + np.random.normal(loc=mean_noize, scale=deviation_noize)

        K = prev_P / (prev_P + dev_r)
        P_[-1] = dev_r * prev_P / (prev_P + dev_r)
        x_[-1] = prev_y + K * (observed_y[-1] - prev_y)

        lines1.set_data(x, y)
        lines2.set_data(x, observed_y)
        lines3.set_data(x, x_)
        ax.set_xlim((x.min(), x.max()))
        ax.set_ylim((observed_y.min()-5, observed_y.max()+5))

        plt.pause(0.01)


    #kf = Simple_Kalman(observed_position, start_position=0, start_deviation=1.0, deviation_true=1.0, deviation_noise=10.0)
