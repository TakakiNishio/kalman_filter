# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt


def pause_plot():

    fig, ax = plt.subplots(1, 1)
    x = np.arange(-np.pi, np.pi, 0.1)
    y = np.sin(x)
    lines, = ax.plot(x, y)

    while True:

        x += 0.1
        y = np.sin(x)

        lines.set_data(x, y)

        ax.set_xlim((x.min(), x.max()))

        plt.pause(.01)

if __name__ == "__main__":

    pause_plot()
