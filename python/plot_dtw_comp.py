import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sen_type', choices=['active', 'passive'])
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--sen1', type=int, default=6)
    args = parser.parse_args()

    sen_type = args.sen_type
    sen0 = args.sen0
    sen1 = args.sen1

    fname_str = '/share/volume0/nrafidi/DTW/dtw_mat_score_{sen_type}_{sen0}vs{sen1}_{radius}_{dist}.npz'

    euclidean_scores = []
    euclidean_rads = []
    for radius in range(1, 2501, 25):
        fname = fname_str.format(sen_type=sen_type,
                                 sen0=sen0,
                                 sen1=sen1,
                                 radius=radius,
                                 dist='euclidean')
        if not os.path.isfile(fname):
            break
        result = np.load(fname)
        euclidean_scores.append(result['scores'])
        euclidean_rads.append(radius)

    cosine_scores = []
    cosine_rads = []
    for radius in range(1, 2501, 25):
        fname = fname_str.format(sen_type=sen_type,
                                 sen0=sen0,
                                 sen1=sen1,
                                 radius=radius,
                                 dist='cosine')
        if not os.path.isfile(fname):
            break
        result = np.load(fname)
        cosine_scores.append(result['scores'])
        cosine_rads.append(radius)


    fig, ax = plt.subplots()
    ax.plot(euclidean_rads, euclidean_scores, c='r', label='euclidean')
    ax.plot(cosine_rads, cosine_scores, c='b', label='cosine')
    ax.set_xlabel('Radius of Exact DTW')
    ax.set_ylabel('Correlation with Binary Matrix')
    ax.set_title('Correlation between MEG data and Binary Matrix\n{sen_type}'.format(sen_type=sen_type))
    ax.legend()
    plt.savefig(
        '/home/nrafidi/thesis_figs/dtw_comp_{sen_type}_{sen0}vs{sen1}.png'.format(
            sen_type=sen_type,
            sen0=sen0,
            sen1=sen1
        ), bbox_inches='tight')

    best_euclidean = np.argmax(np.array(euclidean_scores))
    best_cosine = np.argmax(np.array(cosine_scores))

    fname = fname_str.format(sen_type=sen_type,
                             sen0=sen0,
                             sen1=sen1,
                             radius=euclidean_rads[best_euclidean],
                             dist='euclidean')
    result = np.load(fname)
    euclid_mat = result['dtw_mat']
    fname = fname_str.format(sen_type=sen_type,
                             sen0=sen0,
                             sen1=sen1,
                             radius=cosine_rads[best_cosine],
                             dist='cosine')
    result = np.load(fname)
    cosine_mat = result['dtw_mat']

    num_sen = 20
    comp_mat = np.empty((num_sen, num_sen))
    comp_mat[:10, :10] = 0.0
    comp_mat[10:, 10:] = 0.0
    comp_mat[:10, 10:] = 1.0
    comp_mat[10:, :10] = 1.0

    fig, axs = plt.subplots(1,3)
    h0 = axs[0].imshow(euclid_mat, interpolation='nearest')
    fig.colorbar(h0, ax=axs[0], shrink=0.5)
    axs[0].set_title('Euclidean')
    h1 = axs[1].imshow(cosine_mat, interpolation='nearest')
    fig.colorbar(h1, ax=axs[1], shrink=0.5)
    axs[1].set_title('Cosine')
    h2 = axs[2].imshow(comp_mat, interpolation='nearest')
    fig.colorbar(h2, ax=axs[1], shrink=0.5)
    axs[2].set_title('True')
    fig.suptitle('Repetition-Repetition RDMs\n{sen_type}'.format(sen_type=sen_type))
    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/dtw_rdm_comp_{sen_type}_{sen0}vs{sen1}.png'.format(
            sen_type=sen_type,
            sen0=sen0,
            sen1=sen1
        ), bbox_inches='tight')

    plt.show()