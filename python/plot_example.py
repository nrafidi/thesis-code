import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    x = np.linspace(0.0, 20.0, 1000)

    sig = np.sin(x) + np.sin(5*x) + np.sin(10*x)

    fig, ax = plt.subplots()
    ax.plot(x, sig, color='r', linewidth=5.0)

    colors = ['b', 'g']
    for i_win, win_len in enumerate([0.5, 2.0]):
        win_starts = np.arange(0.0, 20.0, win_len)
        time_wins = [np.logical_and(x >= win, x <= (win + win_len)) for win in win_starts]

        avg_sig = np.array([np.mean(sig[tm]) for tm in time_wins]) - 3.5 - 3.0*i_win
        subsamp = len(x)/len(avg_sig)
        ax.plot(x[::subsamp], avg_sig, color=colors[i_win], linewidth=5.0)


    meg_sig_class1_rep1 = np.sin(x) + np.sin(0.5*x) + np.sin(2.0*x) + np.sin(50.0*x) + np.random.randn(1000)
    meg_sig_class1_rep2 = np.sin(x) + np.sin(0.5 * x) + np.sin(2.0 * x) + np.sin(50.0 * x) + np.random.randn(1000)

    meg_sig_class2_rep1 = np.sin(x) + np.sin(0.1 * x) + np.sin(2.5 * x) + np.sin(50.0 * x) + np.random.randn(1000)
    meg_sig_class2_rep2 = np.sin(x) + np.sin(0.1 * x) + np.sin(2.5 * x) + np.sin(50.0 * x) + np.random.randn(1000)

    meg_sig_class1_rep3 = np.sin(x) + np.sin(0.5 * x) + np.sin(2.0 * x) + np.sin(50.0 * x) + np.random.randn(1000)
    meg_sig_class1_rep4 = np.sin(x) + np.sin(0.5 * x) + np.sin(2.0 * x) + np.sin(50.0 * x) + np.random.randn(1000)

    fig, ax = plt.subplots()
    ax.plot(x, meg_sig_class1_rep1, color='b')
    ax.plot(x, meg_sig_class1_rep2 + 4.0, color='b')
    ax.plot(x, meg_sig_class2_rep2 + 8.0, color='r')
    ax.plot(x, meg_sig_class2_rep1 + 12.0, color='r')

    fig, ax = plt.subplots()
    ax.plot(x, meg_sig_class1_rep3, color='b')
    ax.plot(x, meg_sig_class1_rep4 + 4.0, color='b')

    fig, ax = plt.subplots()
    ax.plot(x, (meg_sig_class1_rep3 + meg_sig_class1_rep4)/2.0, color='b')

    plt.show()
