#!python
#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    frames = []
    for dx in np.arange(1, 10, 0.1):
        x = np.linspace(0, 5, 100)
        y = np.sin(x + dx)
        tmp_frame, = ax.plot(x, y, "b-")
        frames.append([tmp_frame])

    ax.axis([0, 5, -1.1, 1.1])
    ani = anim.ArtistAnimation(fig, frames, interval=1, repeat=True, repeat_delay=1000)

    plt.show(block=False)
    input("Enter to close")
    plt.close()

    # ani.save("test.mp4", writer="ffmpeg", fps=30, bitrate=1000)

    print("end")

if __name__ == '__main__':
    main()
