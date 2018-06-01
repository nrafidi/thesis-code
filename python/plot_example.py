import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


if __name__ == '__main__':

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

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


    meg_sig_class1_rep1 = np.sin(x) + np.sin(0.5*x) + np.sin(2.0*x) + np.random.randn(1000)
    meg_sig_class1_rep2 = np.sin(x) + np.sin(0.5 * x) + np.sin(2.0 * x) + np.random.randn(1000)

    meg_sig_class2_rep1 = np.sin(x) + np.sin(0.1 * x) + np.sin(3.5 * x) + np.random.randn(1000)
    meg_sig_class2_rep2 = np.sin(x) + np.sin(0.1 * x) + np.sin(3.5 * x) + np.random.randn(1000)

    meg_sig_class1_rep3 = np.sin(x) + np.sin(0.5 * x) + np.sin(2.0 * x) + np.random.randn(1000)
    meg_sig_class1_rep4 = np.sin(x) + np.sin(0.5 * x) + np.sin(2.0 * x) + np.random.randn(1000)

    fig, ax = plt.subplots()
    ax.plot(x, meg_sig_class1_rep1, color='b')
    ax.plot(x, meg_sig_class1_rep2 + 7.0, color='b')
    ax.plot(x, meg_sig_class2_rep2 + 14.0, color='r')
    ax.plot(x, meg_sig_class2_rep1 + 21.0, color='r')

    fig, ax = plt.subplots()
    ax.plot(x, meg_sig_class1_rep3, color='b')
    ax.plot(x, meg_sig_class1_rep4 + 7.0, color='b')

    fig, ax = plt.subplots()
    ax.plot(x, (meg_sig_class1_rep3 + meg_sig_class1_rep4)/2.0, color='b')

    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), color='b', linewidth=3.0, label='Class A')
    ax.plot(x, np.sin(x + np.pi), color='r', linewidth=3.0, label='Class B')
    ax.set_xlim([0.0, 10.0])
    ax.set_xlabel('Time', fontsize=axislabelsize)
    ax.set_ylabel('Signal', fontsize=axislabelsize)
    ax.legend(fontsize=legendfontsize)
    rect = patches.Rectangle((1.5, -0.8), width=3.25, height=1.6, fill=False, edgecolor='g', linewidth=3.0)
    ax.add_patch(rect)
    fig.savefig(
        '/home/nrafidi/thesis_figs/time_avg_example.pdf', bbox_inches='tight')
    fig.savefig(
        '/home/nrafidi/thesis_figs/time_avg_example.png', bbox_inches='tight')

    fig, axs = plt.subplots(ncols=1, nrows=2)
    ax = axs[0]
    ax.plot(x, np.sin(x), color='b', linewidth=3.0)
    ax.set_xlim([0.0, 10.0])
    # ax.set_xlabel('Time', fontsize=axislabelsize)
    # ax.set_ylabel('Signal', fontsize=axislabelsize)
    # ax.legend(fontsize=legendfontsize)
    rect = patches.Rectangle((1.5, -0.8), width=4.25, height=1.6, fill=False, edgecolor='g', linewidth=3.0)
    rect = patches.Rectangle((2.0, -0.8), width=4.25, height=1.6, fill=False, edgecolor='r', linewidth=3.0)
    ax.add_patch(rect)
    ax = axs[1]
    ind_to_plot_0 = np.logical_and(x > 1.5, x < 5.75)
    ind_to_plot_1 = np.logical_and(x > 2.0, x < 6.25)
    ax.plot(x[ind_to_plot_0], np.sin(x[ind_to_plot_0]), color='g', linewidth=3.0)
    ax.plot(x[ind_to_plot_1], np.sin(x[ind_to_plot_1]), color='r', linewidth=3.0)
    # ax.set_xlim([0.0, 10.0])
    ax.set_xlabel('Time', fontsize=axislabelsize)
    # ax.set_ylabel('Signal', fontsize=axislabelsize)
    fig.text(0.04, 0.275, 'Signal', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    fig.savefig(
        '/home/nrafidi/thesis_figs/tgm_example.pdf', bbox_inches='tight')
    fig.savefig(
        '/home/nrafidi/thesis_figs/tgm_example.png', bbox_inches='tight')



    plt.show()
