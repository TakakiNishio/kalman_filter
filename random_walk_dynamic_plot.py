#!python
#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

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

    start_position = 0.0
    position = start_position

    fig, ax = plt.subplots(1, 1)
    x = np.arange(0, 1, 0.1)
    y = np.random.normal(loc=0.0, scale=1.0, size=len(x))

    x = x.tolist()
    y = y.tolist()

    print type(y)

    print len(x)
    print len(y)
    lines, = ax.plot(x, y)

    k = 0
    while True:

        k+=1
        move = np.random.normal(loc=0.0, scale=1.0)
        x.append(x[-1]+0.1)
        y.append(y[-1]+move)

        print len(x)
        print len(y)
        lines.set_data(x, y)

        ax.set_xlim((-0.1+0.1*k, x[-1]+0.1))
        ax.set_ylim((-5, 5))

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
