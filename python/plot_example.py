import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    x = np.linspace(0.0, 20.0, 1000)

    sig = np.sin(x) + np.sin(3*x) + np.sin(10*x)

    win_starts = np.arange(0.0, 20.0, 2.0)
    time_wins = [np.logical_and(x >= win, x <= (win + 2.0)) for win in win_starts]

    avg_sig = np.array([np.mean(sig[tm]) for tm in time_wins])


    fig, ax = plt.subplots()
    ax.plot(sig)
    ax.plot(avg_sig)


    plt.show()