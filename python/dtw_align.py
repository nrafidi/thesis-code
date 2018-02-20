import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
import load_data_ordered as load_data
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import kendalltau
import fastdtw


def apply_dtw(sen_data0, sen_data1, radius, dist, sensors):
    if sensors == 'all':
        dtw, path = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data0)),
                                    np.transpose(np.squeeze(sen_data1)),
                                    radius=radius,
                                    dist=dist)
    elif sensors == 'separate':
        dtw = 0.0
        sensor_paths = []
        for i_sensor in range(sen_data0.shape[0]):
            curr_dist, curr_path = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data0[i_sensor, :])),
                                                   np.transpose(np.squeeze(sen_data1[i_sensor, :])),
                                                   radius=radius,
                                                   dist=dist)
            dtw += curr_dist
            sensor_paths.append(curr_path[None, ...])
        path = np.concatenate(sensor_paths, axis=0)
    elif sensors == 'three':
        dtw = 0.0
        sensor_paths = []
        for i_sensor in range(0, sen_data0.shape[0], 3):
            curr_dist, curr_path = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data0[i_sensor:(i_sensor + 3), :])),
                                                   np.transpose(np.squeeze(sen_data1[i_sensor:(i_sensor + 3), :])),
                                                   radius=radius,
                                                   dist=dist)
            dtw += curr_dist
            sensor_paths.append(curr_path[None, ...])
        path = np.concatenate(sensor_paths, axis=0)
    else:
        dtw, path = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data0[2::3, :])),
                                    np.transpose(np.squeeze(sen_data1[2::3, :])),
                                    radius=radius,
                                    dist=dist)
    return dtw, path


def noalign_dist(sen_data0, sen_data1, dist, sensors):
    if sensors == 'all':
        na_dist = np.sum([dist(np.squeeze(sen_data0[:, i]),
                       np.squeeze(sen_data1[:, i])) for i in range(sen_data.shape[-1])])
    elif sensors == 'separate':
        na_dist = 0.0
        for i_sensor in range(sen_data0.shape[0]):
            na_dist += np.sum([dist(np.squeeze(sen_data0[i_sensor, i]),
                                   np.squeeze(sen_data1[i_sensor, i])) for i in range(sen_data.shape[-1])])
    elif sensors == 'three':
        na_dist = 0.0
        for i_sensor in range(0, sen_data0.shape[0]):
            na_dist += np.sum([dist(np.squeeze(sen_data0[i_sensor:(i_sensor+3), i]),
                                    np.squeeze(sen_data1[i_sensor:(i_sensor+3), i])) for i in range(sen_data.shape[-1])])
    else:
        na_dist = np.sum([dist(np.squeeze(sen_data0[2::3, i]),
                               np.squeeze(sen_data1[2::3, i])) for i in range(sen_data.shape[-1])])
    return na_dist


def warp_data(sen_data, path, sensors):
    if sensors == 'all' or sensors == 'mag':
        warp_sen_data = np.transpose(np.squeeze(sen_data[:, path]))
    elif sensors == 'separate':
        warp_sen_data = np.empty((sen_data.shape[0], path.shape[1]))
        for i_sensor in range(path.shape[0]):
            warp_sen_data[i_sensor, :] = np.squeeze(sen_data[i_sensor, path[i_sensor, :]])
    else:
        warp_sen_data = np.empty((sen_data.shape[0], path.shape[1]))
        for i_group in range(0, path.shape[0]):
            i_sensor = i_group*3
            warp_sen_data[i_sensor:(i_sensor+3), :] = np.squeeze(sen_data[i_sensor:(i_sensor+3), path[i_group, :]])
    return warp_sen_data


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
    parser.add_argument('--rep', type=int)
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--sen1', type=int, default=6)
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tmin', type=float, default=0.3)
    parser.add_argument('--sensors', choices=['all', 'separate', 'three', 'mag'])
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
    rep0 = args.rep
    tmin = args.tmin
    tmax = args.tmax
    sensors = args.sensors

    if rep0 < num_instances - 1:
        rep1 = rep0 + 1
    else:
        rep1 = rep0 - 1

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

    dtw_within, path_within = apply_dtw(sen_data[rep0, :, :], sen_data[rep1, :, :], radius, dist, sensors)

    print('Within sentence dtw distance: {}'.format(dtw_within))
    path_within = np.array(path_within)

    dtw_without, path_without = apply_dtw(sen_data[rep0, :, :], sen_data[rep0 + num_instances, :, :], radius, dist, sensors)

    path_without = np.array(path_without)
    print('Across sentence dtw distance: {}'.format(dtw_without))

    dist_noalign_within = noalign_dist(sen_data[rep0, :, :], sen_data[rep1, :, :], dist, sensors)

    print('Within sentence no align distance: {}'.format(dist_noalign_within))

    dist_noalign_without = noalign_dist(sen_data[rep0, :, :], sen_data[rep0+num_instances, :, :], dist, sensors)

    print('Across sentence no align distance: {}'.format(dist_noalign_without))

    orig_rep0_data = np.squeeze(sen_data[rep0, :, :])
    orig_rep0_data /= np.max(np.abs(orig_rep0_data))

    warp_rep0_data = warp_data(sen_data[rep0, :, :], path_within[..., 0], sensors)
    warp_rep0_data /= np.max(np.abs(warp_rep0_data))

    orig_rep1_data = np.squeeze(sen_data[rep1, :, :])
    orig_rep1_data /= np.max(np.abs(orig_rep1_data))

    warp_rep1_data = warp_data(sen_data[rep1, :, :], path_within[..., 1], sensors)
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
    fig.suptitle('Within Sentence\nDTW: {} No Align: {}'.format(dtw_within, dist_noalign_within), fontsize=18)
    plt.subplots_adjust(top=0.8)
    plt.savefig('/home/nrafidi/thesis_figs/dtw_rep_within_ni{}_{}_r{}_{}.png'.format(num_instances,
                                                                                     sen_type,
                                                                                     radius,
                                                                                     args.dist),
                bbox_inches='tight')

    orig_sen0_data = np.squeeze(sen_data[rep0, :, :])
    orig_sen0_data /= np.max(np.abs(orig_sen0_data))

    warp_sen0_data = warp_data(sen_data[rep0, :, :], path_without[..., 0], sensors)
    warp_sen0_data /= np.max(np.abs(warp_sen0_data))

    orig_sen1_data = np.squeeze(sen_data[num_instances, :, :])
    orig_sen1_data /= np.max(np.abs(orig_sen1_data))

    warp_sen1_data = warp_data(sen_data[rep0+num_instances, :, :], path_within[..., 1], sensors)
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
    fig.suptitle('Across Sentence\nDTW: {} No Align: {}'.format(dtw_without, dist_noalign_without), fontsize=18)

    plt.subplots_adjust(top=0.8)
    plt.savefig('/home/nrafidi/thesis_figs/dtw_rep_without_ni{}_{}_r{}_{}.png'.format(num_instances,
                                                                                      sen_type,
                                                                                      radius,
                                                                                      args.dist),
                bbox_inches='tight')
    plt.show()