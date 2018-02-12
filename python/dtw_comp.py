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
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--sen1', type=int, default=6)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)

    args = parser.parse_args()

    exp = args.experiment
    sub = args.subject
    proc = args.proc
    radius = args.radius
    sen0 = args.sen0
    sen1 = args.sen1
    sen_type = args.sen_type

    if args.dist == 'euclidean':
        dist=euclidean
    else:
        dist=cosine

    data, labels, time, final_inds = load_data.load_sentence_data(subject=sub,
                                                                  word='noun1',
                                                                  sen_type=sen_type,
                                                                  experiment=exp,
                                                                  proc=proc,
                                                                  num_instances=10,
                                                                  reps_to_use=10,
                                                                  noMag=False,
                                                                  sorted_inds=None)

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
    num_sen = 20


    comp_mat = np.empty((num_sen, num_sen))
    comp_mat[:10, :10] = 0.0
    comp_mat[10:, 10:] = 0.0
    comp_mat[:10, 10:] = 1.0
    comp_mat[10:, :10] = 1.0
    # print(comp_mat)

    radius_range = range(1, data.shape[-1], 12)

    dtw_mat = np.empty((num_sen, num_sen))
    for i in range(num_sen):
        for j in range(i, num_sen):
            dtw_mat[i, j], _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[i, :, :])),
                                               np.transpose(np.squeeze(sen_data[j, :, :])),
                                               radius=radius,
                                               dist=dist)
            dtw_mat[j, i] = dtw_mat[i, j]
    score, _ = ktau_rdms(comp_mat, dtw_mat)
    print('Score {} at radius {} distance {}'.format(score, radius, args.dist))

    np.savez('/share/volume0/nrafidi/DTW/dtw_mat_score_{sen_type}_{sen0}vs{sen1}_{radius}_{dist}.npz'.format(sen_type=sen_type,
                                                                                                             sen0=sen0,
                                                                                                             sen1=sen1,
                                                                                                             radius=radius,
                                                                                                             dist=dist),
             scores=score, dtw_mat = dtw_mat)
