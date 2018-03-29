import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import os
from scipy.stats import spearmanr, kendalltau

RESULT_FNAME = '/share/volume0/nrafidi/DTW/EOS_dtw_sensor{i_sensor}_score_{exp}_{sub}_sen{sen0}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'
SCORE_FNAME = '/share/volume0/nrafidi/DTW/EOS_dtw_sensor{i_sensor}_score_{exp}_{sub}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'


def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    upper_tri_inds = np.triu_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[upper_tri_inds],rdm2[upper_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--dist', choices=['euclidean', 'cosine'], default='cosine')
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--num_instances', type=int, default=10)
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tmax', type=float, default=3.0)

    args = parser.parse_args()

    exp = args.experiment
    sub = args.subject
    radius = args.radius
    num_instances = args.num_instances
    tmin = args.tmin
    tmax = args.tmax

    if args.dist == 'euclidean':
        dist=euclidean
    else:
        dist=cosine

    comp_rdm = np.ones((num_instances*32, num_instances*32), dtype=float)
    total_rdm = np.empty((306, num_instances*32, num_instances*32), dtype='float')
    for sen0 in range(32):
        start_ind = sen0*num_instances
        end_ind = start_ind + num_instances
        comp_rdm[start_ind:end_ind, start_ind:end_ind] = 0.0
        for i_sensor in range(306):
            result_fname = RESULT_FNAME.format(exp=exp, sub=sub, sen0=sen0, radius=radius, dist=args.dist,
                                         ni=num_instances, tmin=tmin, tmax=tmax, i_sensor=i_sensor)
            if not os.path.isfile(result_fname):
                print(result_fname)
                break
            result = np.load(result_fname)
            dtw_part = result['dtw_part']
            other_sens = range(dtw_part.shape[0])
            for sen1 in other_sens:
                start_ind_y = start_ind + sen1*num_instances
                end_ind_y = start_ind_y + num_instances 
                total_rdm[i_sensor, start_ind:end_ind, start_ind_y:end_ind_y] = dtw_part[sen1, :, :]
                total_rdm[i_sensor, start_ind_y:end_ind_y, start_ind:end_ind] = dtw_part[sen1, :, :]

    fig, ax = plt.subplots()
    ax.imshow(comp_rdm, interpolation='nearest')
    ax.set_title('Model RDM')
    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/EOS_model-rdm_ni{ni}.png'.format(
            ni=num_instances),
        bbox_inches='tight')

    scores = np.empty((306,))
    for i_sensor in range(306):
        print(i_sensor)
        score_fname = SCORE_FNAME.format(exp=exp, sub=sub, radius=radius, dist=args.dist,
                                           ni=num_instances, tmin=tmin, tmax=tmax, i_sensor=i_sensor)
        # if os.path.isfile(score_fname):
        #     result = np.load(score_fname)
        #     scores[i_sensor] = result['sensor_score']
        # else:
        scores[i_sensor], _ = ktau_rdms(total_rdm[i_sensor, :, :], comp_rdm)
        np.savez(score_fname, sensor_score=scores[i_sensor])

    fig, ax = plt.subplots()
    ax.hist(scores)
    ax.set_title('Histogram of Sensor Correlations with Model\nNumber of instances: {ni}'.format(ni=num_instances))
    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/EOS_score-hist_{exp}_{sub}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.png'.format(
            exp=exp,
            sub=sub,
            radius=radius,
            dist=args.dist,
            ni=num_instances,
            tmin=tmin,
            tmax=tmax),
        bbox_inches='tight')

    best_sensor = np.argmax(scores)
    print(best_sensor)
    fig, ax = plt.subplots()
    h = ax.imshow(np.squeeze(total_rdm[best_sensor, :, :]), interpolation='nearest', vmin=0.0)
    ax.set_title('Best Sensor')
    plt.colorbar(h)
    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/EOS_best-sensor_{exp}_{sub}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.png'.format(
            exp=exp,
            sub=sub,
            radius=radius,
            dist=args.dist,
            ni=num_instances,
            tmin=tmin,
            tmax=tmax),
        bbox_inches='tight')

    mean_sensor = np.mean(total_rdm, axis=0)
    fig, ax = plt.subplots()
    h = ax.imshow(np.squeeze(mean_sensor), interpolation='nearest', vmin=0.0)
    ax.set_title('Average over Sensors')
    plt.colorbar(h)
    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/EOS_avg-sensor_{exp}_{sub}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.png'.format(
            exp=exp,
            sub=sub,
            radius=radius,
            dist=args.dist,
            ni=num_instances,
            tmin=tmin,
            tmax=tmax),
        bbox_inches='tight')

    mean_score = ktau_rdms(mean_sensor, comp_rdm)
    print('Score of mean over sensors: {}'.format(mean_score))
    plt.show()
