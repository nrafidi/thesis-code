import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sen_type', choices=['active', 'passive'])
    parser.add_argument('--dist', choices=['euclidean', 'cosine'], default='cosine')
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--sen1', type=int, default=6)
    # parser.add_argument('--num_instances', type=int)
    # parser.add_argument('--sensors', choices=['all', 'separate', 'three', 'mag'])
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tmax', type=float, default=0.3)
    args = parser.parse_args()

    sen_type = args.sen_type
    sen0 = args.sen0
    sen1 = args.sen1
    # num_instances = args.num_instances
    # sensors = args.sensors
    dist = args.dist
    tmin = args.tmin
    tmax = args.tmax


    fname_str = '/share/volume0/nrafidi/DTW/dtw_mat_score_{sen_type}_{sen0}vs{sen1}_{radius}_{dist}_{sensors}_ni{ni}_{tmin}-{tmax}.npz'
    # fname_str = '/share/volume0/nrafidi/DTW/dtw_mat_score_{sen_type}_{sen0}vs{sen1}_{radius}_{dist}.npz'

    scores = []
    for radius in range(1, 151, 25):
        scores_by_ni = []
        for ni in [2, 5, 10]:
            scores_by_sens = []
            for sens in ['all', 'separate', 'three', 'mag']:
                fname = fname_str.format(sen_type=sen_type,
                                         sen0=sen0,
                                         sen1=sen1,
                                         radius=radius,
                                         dist=dist,
                                         sensors=sens,
                                         ni=ni,
                                         tmin=tmin,
                                         tmax=tmax)
                result = np.load(fname)
                score_mat = result['scores']
                scores_by_sens.append(score_mat[None, ...])
            scores_by_sens = np.concatenate(scores_by_sens, axis=0)
            scores_by_ni.append(scores_by_sens[None, ...])
        scores_by_ni = np.concatenate(scores_by_ni, axis=0)
        scores.append(scores_by_ni[None, ...])
    scores = np.concatenate(scores, axis=0)
    print(scores)
    max_by_rad = np.max(scores, axis=0)
    best_rad = np.argmax(scores, axis=0)
    print(best_rad)

    fig, ax = plt.subplots()
    h = ax.imshow(max_by_rad, interpolation='nearest')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['all', 'separate', 'three', 'mag'])
    ax.set_xlabel('Sensor Treatment')
    ax.set_yticks(range(3))
    ax.set_yticklabels([2, 5, 10])
    ax.set_ylabel('Number of Instances')
    ax.set_title('Correlation between MEG and Binary Matrix\n{sen_type} {tmin}-{tmax}'.format(sen_type=sen_type,
                                                                                              tmin=tmin,
                                                                                              tmax=tmax))
    plt.colorbar(h)
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(euclidean_rads, euclidean_scores, c='r', label='euclidean')
    # ax.plot(cosine_rads, cosine_scores, c='b', label='cosine')
    # ax.set_xlabel('Radius of Exact DTW')
    # ax.set_ylabel('Correlation with Binary Matrix')
    # ax.set_title('Correlation between MEG data and Binary Matrix\n{sen_type}'.format(sen_type=sen_type))
    # ax.legend()
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/dtw_comp_{sen_type}_{sen0}vs{sen1}.png'.format(
    #         sen_type=sen_type,
    #         sen0=sen0,
    #         sen1=sen1
    #     ), bbox_inches='tight')
    #
    # best_euclidean = np.argmax(np.array(euclidean_scores))
    # best_cosine = np.argmax(np.array(cosine_scores))
    #
    # fname = fname_str.format(sen_type=sen_type,
    #                          sen0=sen0,
    #                          sen1=sen1,
    #                          radius=euclidean_rads[best_euclidean],
    #                          dist='euclidean')
    # result = np.load(fname)
    # euclid_mat = result['dtw_mat']
    # fname = fname_str.format(sen_type=sen_type,
    #                          sen0=sen0,
    #                          sen1=sen1,
    #                          radius=cosine_rads[best_cosine],
    #                          dist='cosine')
    # result = np.load(fname)
    # cosine_mat = result['dtw_mat']
    #
    # num_sen = 20
    # comp_mat = np.empty((num_sen, num_sen))
    # comp_mat[:10, :10] = 0.0
    # comp_mat[10:, 10:] = 0.0
    # comp_mat[:10, 10:] = 1.0
    # comp_mat[10:, :10] = 1.0
    #
    # fig, axs = plt.subplots(1,3)
    # h0 = axs[0].imshow(euclid_mat, interpolation='nearest') #, aspect='auto')
    # fig.colorbar(h0, ax=axs[0], shrink=0.5)
    # axs[0].set_title('Euclidean')
    # h1 = axs[1].imshow(cosine_mat, interpolation='nearest') #, aspect='auto')
    # fig.colorbar(h1, ax=axs[1], shrink=0.5)
    # axs[1].set_title('Cosine')
    # h2 = axs[2].imshow(comp_mat, interpolation='nearest') #, aspect='auto')
    # fig.colorbar(h2, ax=axs[2], shrink=0.5)
    # axs[2].set_title('True')
    # fig.suptitle('Repetition-Repetition RDMs\n{sen_type}'.format(sen_type=sen_type))
    # fig.tight_layout()
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/dtw_rdm_comp_{sen_type}_{sen0}vs{sen1}.png'.format(
    #         sen_type=sen_type,
    #         sen0=sen0,
    #         sen1=sen1
    #     ), bbox_inches='tight')

    # plt.show()