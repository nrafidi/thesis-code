import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import os
from scipy.stats import spearmanr, kendalltau

RESULT_FNAME = '/share/volume0/nrafidi/DTW/EOS_dtw_sensor{i_sensor}_score_{exp}_{sub}_sen{sen0}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'


def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    lower_tri_inds = np.tril_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[lower_tri_inds],rdm2[lower_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--dist', choices=['euclidean', 'cosine'])
    parser.add_argument('--radius', type=int)
    parser.add_argument('--num_instances', type=int)
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tmax', type=float, default=1.0)

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

    fig, ax = plt.subplots()
    ax.imshow(comp_rdm, interpolation='nearest')
    ax.set_title('Model RDM')

    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(total_rdm[2, :, :]), interpolation='nearest')
    
    plt.show()

    scores = np.empty((306,))
    for i_sensor in range(306):
        print(i_sensor)
        scores[i_sensor], _ = ktau_rdms(total_rdm[i_sensor, :, :], comp_rdm)

    fig, ax = plt.subplots()
    ax.hist(scores)
    ax.set_title('Score Histogram\n{exp} {sub} {dist}\nEOS {tmin}-{tmax} ni {ni}'.format(exp=exp,
                                                                                         sub=sub,
                                                                                         dist=args.dist,
                                                                                         tmin=tmin,
                                                                                         tmax=tmax,
                                                                                         ni=num_instances))

    plt.show()
