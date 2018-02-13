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

    dtw_within, path_within = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[0, :, :])),
                                              np.transpose(np.squeeze(sen_data[1, :, :])),
                                              radius=radius,
                                              dist=dist)

    print(dtw_within)
    path_within = np.array(path_within)
    print(path_within.shape)

    dtw_without, path_without = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[0, :, :])),
                                                np.transpose(np.squeeze(sen_data[10, :, :])),
                                                radius=radius,
                                                dist=dist)
    print(dtw_without)

    dist_noalign_within = np.sum([cosine(np.squeeze(sen_data[0, :, i]),
                                  np.squeeze(sen_data[1, :, i])) for i in range(sen_data.shape[-1])])

    print(dist_noalign_within)

    dist_noalign_without = np.sum([cosine(np.squeeze(sen_data[0, :, i]),
                                         np.squeeze(sen_data[10, :, i])) for i in range(sen_data.shape[-1])])

    print(dist_noalign_without)