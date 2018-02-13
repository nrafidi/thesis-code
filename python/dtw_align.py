import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
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
    num_instances=args.num_instances

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

    print('Within sentence dtw distance: {}'.format(dtw_within))
    path_within = np.array(path_within)
    print(path_within.shape)

    dtw_without, path_without = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[0, :, :])),
                                                np.transpose(np.squeeze(sen_data[num_instances, :, :])),
                                                radius=radius,
                                                dist=dist)
    print('Across sentence dtw distance: {}'.format(dtw_without))

    dist_noalign_within = np.sum([cosine(np.squeeze(sen_data[0, :, i]),
                                  np.squeeze(sen_data[1, :, i])) for i in range(sen_data.shape[-1])])

    print('Within sentence no align distance: {}'.format(dist_noalign_within))

    dist_noalign_without = np.sum([cosine(np.squeeze(sen_data[0, :, i]),
                                         np.squeeze(sen_data[num_instances, :, i])) for i in range(sen_data.shape[-1])])

    print('Across sentence no align distance: {}'.format(dist_noalign_without))

    orig_rep0_data = np.squeeze(sen_data[0, :, :])
    orig_rep0_data /= np.max(np.abs(orig_rep0_data))

    warp_rep0_data = np.transpose(np.squeeze(sen_data[0, :, path_within[:,0]]))
    warp_rep0_data /= np.max(np.abs(warp_rep0_data))

    orig_rep1_data = np.squeeze(sen_data[1, :, :])
    orig_rep1_data /= np.max(np.abs(orig_rep1_data))

    warp_rep1_data = np.transpose(np.squeeze(sen_data[1, :, path_within[:, 1]]))
    warp_rep1_data /= np.max(np.abs(warp_rep1_data))

    # print(np.sum(np.equal(warp_rep0_data, orig_rep0_data)))
    fig, axs = plt.subplots(2, 2)
    h00 = axs[0][0].imshow(orig_rep0_data, interpolation='nearest', aspect='auto')
    axs[0][0].set_title('Original Sen 0 Rep 0')
    h01 = axs[0][1].imshow(warp_rep0_data, interpolation='nearest', aspect='auto')
    axs[0][1].set_title('Warped Sen 0 Rep 0')
    h10 = axs[1][0].imshow(orig_rep1_data, interpolation='nearest', aspect='auto')
    axs[1][0].set_title('Original Sen 0 Rep 1')
    h11 = axs[1][1].imshow(warp_rep1_data, interpolation='nearest', aspect='auto')
    axs[1][1].set_title('Warped Sen 0 Rep 1')
    fig.suptitle('Within Sentence\nDTW: {} No Align: {}'.format(dtw_within, dist_noalign_within))
    fig.tight_layout()
    plt.savefig('/home/nrafidi/thesis_figs/dtw_rep_within_ni{}_{}_r{}_{}.png'.format(num_instances,
                                                                                     sen_type,
                                                                                     radius,
                                                                                     args.dist),
                bbox_inches='tight')

    orig_sen0_data = np.squeeze(sen_data[0, :, :])
    orig_sen0_data /= np.max(np.abs(orig_sen0_data))

    warp_sen0_data = np.transpose(np.squeeze(sen_data[0, :, path_without[:, 0]]))
    warp_sen0_data /= np.max(np.abs(warp_sen0_data))

    orig_sen1_data = np.squeeze(sen_data[num_instances, :, :])
    orig_sen1_data /= np.max(np.abs(orig_sen1_data))

    warp_sen1_data = np.transpose(np.squeeze(sen_data[num_instances, :, path_without[:, 1]]))
    warp_sen1_data /= np.max(np.abs(warp_sen1_data))

    # print(np.sum(np.equal(warp_rep0_data, orig_rep0_data)))
    fig, axs = plt.subplots(2, 2)
    h00 = axs[0][0].imshow(orig_sen0_data, interpolation='nearest', aspect='auto')
    axs[0][0].set_title('Original Sen 0 Rep 0')
    h01 = axs[0][1].imshow(warp_sen0_data, interpolation='nearest', aspect='auto')
    axs[0][1].set_title('Warped Sen 0 Rep 0')
    h10 = axs[1][0].imshow(orig_sen1_data, interpolation='nearest', aspect='auto')
    axs[1][0].set_title('Original Sen 1 Rep 0')
    h11 = axs[1][1].imshow(warp_sen1_data, interpolation='nearest', aspect='auto')
    axs[1][1].set_title('Warped Sen 1 Rep 0')
    fig.suptitle('Across Sentence\nDTW: {} No Align: {}'.format(dtw_without, dist_noalign_without))

    fig.tight_layout()
    plt.savefig('/home/nrafidi/thesis_figs/dtw_rep_without_ni{}_{}_r{}_{}.png'.format(num_instances,
                                                                                      sen_type,
                                                                                      radius,
                                                                                      args.dist),
                bbox_inches='tight')
    plt.show()