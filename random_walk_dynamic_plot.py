#!python
#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import seaborn as sns
sns.set_context('poster')


def random_walker(start_position=0, mean=0, deviation=1, n_steps=99, seed=None):

    if seed is not None:
        np.random.seed(seed=seed)

    move = np.random.normal(loc=mean, scale=deviation, size=n_steps)
    position = np.insert(move, 0, start_position)
    position = np.cumsum(position)

    return position

def add_noise(position, mean=0, deviation=10, seed=None):

    if seed is not None:
        np.random.seed(seed=seed)

    n_observation = len(position)
    noise = np.random.normal(loc=mean, scale=deviation, size=n_observation)
    observation = position + noise

    return observation


if __name__ == '__main__':

    start_y = 0.0

    mean_y = 0.0
    deviation_y = 1.0

    mean_noize = 0.0
    deviation_noize = 5

    position = start_y
    x = np.arange(0, 1, 0.1)
    move = np.random.normal(loc=mean_y, scale=deviation_y, size=len(x)-1)
    y = np.insert(move, 0, start_y)
    y = np.cumsum(y)

    noise = np.random.normal(loc=mean_noize, scale=deviation_noize, size=len(y))
    observed_y = y + noise

    print type(y)
    print len(x)
    print len(y)

    fig, ax = plt.subplots(1, 1)
    lines1, = ax.plot(x, y, 'r--', label='true y')
    lines2, = ax.plot(x, observed_y, 'y', label='observed y')
    plt.xlabel('time step')
    plt.ylabel('y')
    plt.legend(loc=1)
    # plt.legend(loc='best')
    plt.tight_layout()

    while True:

        prev_x = x[-1]
        prev_y = y[-1]
        prev_observed_y = observed_y[-1]

        x = np.roll(x, -1)
        y = np.roll(y, -1)
        observed_y = np.roll(observed_y, -1)

        x[-1] = prev_x + 0.1
        y[-1] = prev_y + np.random.normal(loc=mean_y, scale=deviation_y)
        observed_y[-1] = y[-1] + np.random.normal(loc=mean_noize, scale=deviation_noize)

        lines1.set_data(x, y)
        lines2.set_data(x, observed_y)

        ax.set_xlim((x.min(), x.max()))
        ax.set_ylim((observed_y.min()-5, observed_y.max()+5))

        plt.pause(.05)

    # true_position = random_walker(start_position=0, mean=0, deviation=1, n_steps=99, seed=0)
    # observed_position = add_noise(true_position, mean=0, deviation=10, seed=0)

    # plt.plot(true_position, 'r--', label='True Positions')
    # plt.plot(observed_position, 'y', label='Observed Positions')
    # # plt.title('Random Walk')
    # plt.xlabel('time step')
    # plt.ylabel('position')
    # plt.legend(loc='best')

    # plt.show()
