import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
from scipy.stats import spearmanr, kendalltau

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'
RESULT_FNAME = '/share/volume0/nrafidi/DTW/EOS_dtw_sensor_score_{exp}_{sub}_{sen0}vs{sen1}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


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

    comp_mat = np.empty((2*num_instances, 2*num_instances))
    comp_mat[:num_instances, :num_instances] = 0.0
    comp_mat[num_instances:, num_instances:] = 0.0
    comp_mat[:num_instances, num_instances:] = 1.0
    comp_mat[num_instances:, :num_instances] = 1.0
    fig, ax = plt.subplots()
    ax.imshow(comp_mat, interpolation='nearest')
    ax.set_title('Ideal RDM')
    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/EOS_model_rdm_ni{ni}.png'.format(
            ni=num_instances),
        bbox_inches='tight')

    for sub in ['B', 'C']:
        result = np.load(RESULT_FNAME.format(exp=exp, sub=sub, sen0=sen0, sen1=sen1, radius=radius, dist=args.dist,
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
        fig.tight_layout()
        plt.savefig(
            '/home/nrafidi/thesis_figs/EOS_score-hist_{exp}_{sub}_{sen0}vs{sen1}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.png'.format(
                exp=exp,
                sub=sub,
                sen0=sen0,
                sen1=sen1,
                radius=radius,
                dist=args.dist,
                ni=num_instances,
                tmin=tmin,
                tmax=tmax),
            bbox_inches='tight')

        best_sens = np.argmax(score_mat)
        worst_sens = np.argmin(score_mat)
        best_mat = np.squeeze(dtw_mat[best_sens, :, :])
        worst_mat = np.squeeze(dtw_mat[worst_sens, :, :])

        fig, ax = plt.subplots()
        ax.imshow(best_mat/np.max(best_mat), interpolation='nearest')
        score_str = '{0:.2f}'.format(np.max(score_mat))
        ax.set_title('Best Sensor score: {score}\n{ni} Instances'.format(score=score_str,
                                                                         ni=num_instances))
        fig.tight_layout()
        plt.savefig(
            '/home/nrafidi/thesis_figs/EOS_best-sensor_{exp}_{sub}_{sen0}vs{sen1}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.png'.format(exp=exp,
                                                                                                                                   sub=sub,
                                                                                                                                   sen0=sen0,
                                                                                                                                   sen1=sen1,
                                                                                                                                   radius=radius,
                                                                                                                                   dist=args.dist,
                                                                                                                                   ni=num_instances,
                                                                                                                                   tmin=tmin,
                                                                                                                                   tmax=tmax),
            bbox_inches='tight')

        fig, ax = plt.subplots()
        ax.imshow(worst_mat/np.max(worst_mat), interpolation='nearest')
        score_str = '{0:.2f}'.format(np.min(score_mat))
        ax.set_title('Worst Sensor score: {score}\n{ni} Instances'.format(
            score=score_str,
            ni=num_instances))

        fig.tight_layout()
        plt.savefig(
            '/home/nrafidi/thesis_figs/EOS_worst-sensor_{exp}_{sub}_{sen0}vs{sen1}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.png'.format(
                exp=exp,
                sub=sub,
                sen0=sen0,
                sen1=sen1,
                radius=radius,
                dist=args.dist,
                ni=num_instances,
                tmin=tmin,
                tmax=tmax),
            bbox_inches='tight')
        scores.append(score_mat[None, ...])

    sub_scores = np.concatenate(scores, axis=0)
    sorted_inds, sorted_reg = sort_sensors()
    sorted_sub_scores = sub_scores[:, sorted_inds]
    good_sensors = np.where(np.all(sub_scores > 0.0, axis=0))[0]
    print('Good sensors: {good_sensors}'.format(good_sensors=good_sensors))
    print([sorted_reg[g_sens] for g_sens in good_sensors])

    sub_corr, _ = spearmanr(sub_scores, axis=1)

    print('Correlation between subjects: {sub_corr}'.format(sub_corr=sub_corr))

    fig, ax = plt.subplots()
    ax.scatter(sub_scores[0, :], sub_scores[1, :])
    ax.plot(np.arange(-0.9, 1.0, 0.1), np.arange(-0.9, 1.0, 0.1), color='r')
    ax.set_xlabel('Subject B sensor scores')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylabel('Subject C sensor scores')
    ax.set_ylim([-1.0, 1.0])

    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/EOS_sub-corr_{exp}_{sen0}vs{sen1}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.png'.format(
            exp=exp,
            sen0=sen0,
            sen1=sen1,
            radius=radius,
            dist=args.dist,
            ni=num_instances,
            tmin=tmin,
            tmax=tmax),
        bbox_inches='tight')

    plt.show()