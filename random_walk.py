import numpy as np
import matplotlib.pyplot as plt
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

true_position = random_walker(start_position=0, mean=0, deviation=1, n_steps=99, seed=0)
observed_position = add_noise(true_position, mean=0, deviation=10, seed=0)

plt.plot(true_position, 'r--', label='True Positions')
plt.plot(observed_position, 'y', label='Observed Positions')
plt.title('Random Walk')
plt.xlabel('time step')
plt.ylabel('position')
plt.legend(loc='best')

plt.show()
