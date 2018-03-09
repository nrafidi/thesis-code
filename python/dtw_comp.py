import argparse
import numpy as np
import load_data_ordered as load_data
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import kendalltau
import fastdtw


def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    lower_tri_inds = np.tril_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[lower_tri_inds],rdm2[lower_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--sen_type', choices=['active', 'passive'])
    parser.add_argument('--dist', choices=['euclidean', 'cosine'])
    parser.add_argument('--radius', type=int)
    parser.add_argument('--num_instances', type=int)
    parser.add_argument('--sensors', choices=['all', 'separate', 'three', 'mag'])
    parser.add_argument('--tmin', type=float, default=0.5)
    parser.add_argument('--tmax', type=float, default=0.8)
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--sen1', type=int, default=8)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)

    args = parser.parse_args()

    exp = args.experiment
    sub = args.subject
    proc = args.proc
    radius = args.radius
    num_instances = args.num_instances
    sen0 = args.sen0
    sen1 = args.sen1
    sen_type = args.sen_type
    tmin = args.tmin
    tmax = args.tmax
    sensors = args.sensors

    if args.dist == 'euclidean':
        dist=euclidean
    else:
        dist=cosine

    data, labels, time, final_inds = load_data.load_sentence_data(subject=sub,
                                                                  word='noun1',
                                                                  sen_type=sen_type,
                                                                  experiment=exp,
                                                                  proc=proc,
                                                                  num_instances=num_instances,
                                                                  reps_to_use=10,
                                                                  noMag=False,
                                                                  sorted_inds=None,
                                                                  tmin=tmin,
                                                                  tmax=tmax)

    sen_set = np.unique(labels, axis=0).tolist()
    num_labels = labels.shape[0]
    sen_ints = np.empty((num_labels,))
    for i_l in range(num_labels):
        for j_l, l in enumerate(sen_set):
            if np.all(l == labels[i_l, :]):
                sen_ints[i_l] = j_l
                break

    sen0_data = data[sen_ints == sen0, ...]
    sen1_data = data[sen_ints == sen1, ...]
    sen_data = np.concatenate([sen0_data, sen1_data], axis=0)
    num_sen = num_instances*2


    comp_mat = np.empty((num_sen, num_sen))
    comp_mat[:num_instances, :num_instances] = 0.0
    comp_mat[num_instances:, num_instances:] = 0.0
    comp_mat[:num_instances, num_instances:] = 1.0
    comp_mat[num_instances:, :num_instances] = 1.0
    # print(comp_mat)

    radius_range = range(1, data.shape[-1], 12)

    dtw_mat = np.empty((num_sen, num_sen))
    noalign_mat = np.empty((num_sen, num_sen))
    for i in range(num_sen):
        for j in range(i, num_sen):
            if sensors == 'all':
                dtw_mat[i, j], _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[i, :, :])),
                                                   np.transpose(np.squeeze(sen_data[j, :, :])),
                                                   radius=radius,
                                                   dist=dist)
            elif sensors == 'separate':
                dist_sum = 0.0
                for i_sensor in range(sen_data.shape[1]):
                    curr_dist, _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[i, i_sensor, :])),
                                                       np.transpose(np.squeeze(sen_data[j, i_sensor, :])),
                                                       radius=radius,
                                                       dist=dist)
                    dist_sum += curr_dist
                dtw_mat[i, j] = dist_sum
            elif sensors == 'three':
                dist_sum = 0.0
                for i_sensor in range(0, sen_data.shape[1], 3):
                    curr_dist, _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[i, i_sensor:(i_sensor + 3), :])),
                                                   np.transpose(np.squeeze(sen_data[j, i_sensor:(i_sensor + 3), :])),
                                                   radius=radius,
                                                   dist=dist)
                    dist_sum += curr_dist
                dtw_mat[i, j] = dist_sum
            else:
                dtw_mat[i, j], _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[i, 2::3, :])),
                                                   np.transpose(np.squeeze(sen_data[j, 2::3, :])),
                                                   radius=radius,
                                                   dist=dist)


            dtw_mat[j, i] = dtw_mat[i, j]
    score, _ = ktau_rdms(comp_mat, dtw_mat)
    print('Score {} at radius {} distance {}'.format(score, radius, args.dist))

    fname = '/share/volume0/nrafidi/DTW/dtw_mat_score_{sen_type}_{sen0}vs{sen1}_{radius}_{dist}_{sensors}_ni{ni}_{tmin}-{tmax}.npz'
    np.savez(fname.format(sen_type=sen_type, sen0=sen0, sen1=sen1, radius=radius, dist=args.dist, sensors=sensors,
                          ni=num_instances, tmin=tmin, tmax=tmax),
             scores=score, dtw_mat = dtw_mat)
