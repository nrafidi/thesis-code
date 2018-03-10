import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import dtw_sensor_select as dtw_sens
import argparse
from scipy.stats import spearmanr, kendalltau

def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    lower_tri_inds = np.tril_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[lower_tri_inds],rdm2[lower_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--dist', choices=['euclidean', 'cosine'])
    parser.add_argument('--radius', type=int)
    parser.add_argument('--num_instances', type=int)
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tmax', type=float, default=1.0)
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--sen1', type=int, default=8)

    args = parser.parse_args()

    exp = args.experiment
    radius = args.radius
    num_instances = args.num_instances
    sen0 = args.sen0
    sen1 = args.sen1
    tmin = args.tmin
    tmax = args.tmax

    scores = []

    for sub in ['B', 'C']:
        result = np.load(dtw_sens.RESULT_FNAME.format(exp=exp, sub=sub, sen0=sen0, sen1=sen1, radius=radius, dist=args.dist,
                                                      ni=num_instances, tmin=tmin, tmax=tmax))

        score_mat = result['scores']
        dtw_mat = result['dtw_mat']

        fig, ax = plt.subplots()
        ax.hist(score_mat)
        ax.set_title('Score Histogram\n{exp} {sub} {sen0}vs{sen1}\nEOS {tmin}-{tmax} ni {ni}'.format(exp=exp,
                                                                                                     sub=sub,
                                                                                                     sen0=sen0,
                                                                                                     sen1=sen1,
                                                                                                     tmin=tmin,
                                                                                                     tmax=tmax,
                                                                                                     ni=num_instances))

        best_sens = np.argmax(score_mat)
        worst_sens = np.argmin(score_mat)
        best_mat = np.squeeze(dtw_mat[best_sens, :, :])
        worst_mat = np.squeeze(dtw_mat[worst_sens, :, :])

        fig, ax = plt.subplots()
        ax.imshow(best_mat/np.max(best_mat), interpolation='nearest')
        ax.set_title('Best Sensor score: {score}\n{exp} {sub} {sen0}vs{sen1}\nEOS {tmin}-{tmax} ni {ni}'.format(score=np.max(score_mat),
                                                                                                                exp=exp,
                                                                                                                sub=sub,
                                                                                                                sen0=sen0,
                                                                                                                sen1=sen1,
                                                                                                                tmin=tmin,
                                                                                                                tmax=tmax,
                                                                                                                ni=num_instances))

        fig, ax = plt.subplots()
        ax.imshow(worst_mat/np.max(worst_mat), interpolation='nearest')
        ax.set_title('Worst Sensor score: {score}\n{exp} {sub} {sen0}vs{sen1}\nEOS {tmin}-{tmax} ni {ni}'.format(
            score=np.min(score_mat),
            exp=exp,
            sub=sub,
            sen0=sen0,
            sen1=sen1,
            tmin=tmin,
            tmax=tmax,
            ni=num_instances))

        scores.append(score_mat[None, ...])

    sub_scores = np.concatenate(scores, axis=0)

    good_sensors = np.where(np.all(sub_scores > 0.0, axis=0))
    print('Good sensors: {good_sensors}'.format(good_sensors=good_sensors))

    sub_corr, _ = spearmanr(sub_scores, axis=1)

    print('Correlation between subjects: {sub_corr}'.format(sub_corr=sub_corr))

    fig, ax = plt.subplots()
    ax.scatter(sub_scores[0, :], sub_scores[1, :])
    ax.plot(np.arange(-0.9, 1.0, 0.1), np.arange(-0.9, 1.0, 0.1), color='r')
    ax.set_xlabel('Subject B sensor scores')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylabel('Subject C sensor scores')
    ax.set_ylim([-1.0, 1.0])

    plt.show()