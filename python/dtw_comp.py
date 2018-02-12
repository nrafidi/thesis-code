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
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)

    args = parser.parse_args()

    exp = args.experiment
    sub = args.subject
    proc = args.proc

    data, labels, time, final_inds = load_data.load_sentence_data(subject=sub,
                                                                  word='noun1',
                                                                  sen_type='active',
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

    sen0_data = data[sen_ints == 0, ...]
    sen1_data = data[sen_ints == 4, ...]
    sen_data = np.concatenate([sen0_data, sen1_data], axis=0)
    num_sen = 20


    comp_mat = np.empty((num_sen, num_sen))
    comp_mat[:10, :10] = 0.0
    comp_mat[10:, 10:] = 0.0
    comp_mat[:10, 10:] = 1.0
    comp_mat[10:, :10] = 1.0
    print(comp_mat)

    radius_range = range(1, data.shape[-1], 12)

    scores = np.empty((len(radius_range), 2))
    for i_rad, radius in enumerate(radius_range):
        for i_dist, dist in enumerate([euclidean, cosine]):
            dtw_mat = np.empty((num_sen, num_sen))
            for i in range(num_sen):
                for j in range(i, num_sen):
                    dtw_mat[i, j], _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[i, :, :])),
                                                       np.transpose(np.squeeze(sen_data[j, :, :])),
                                                       radius=radius,
                                                       dist=dist)
                    dtw_mat[j, i] = dtw_mat[i, j]
            scores[i_rad, i_dist], _ = ktau_rdms(comp_mat, dtw_mat)
            print('Score {} at radius {} distance {}'.format(scores[i_rad, i_dist], radius, dist))

    np.savez('/share/volume0/nrafidi/DTW/dtw_scores.npz', scores=scores)
