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
        if radius < 0:
            dtw, path = fastdtw.dtw(np.transpose(np.squeeze(sen_data0)),
                                    np.transpose(np.squeeze(sen_data1)),
                                    dist=dist)
        else:
            dtw, path = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data0)),
                                        np.transpose(np.squeeze(sen_data1)),
                                        radius=radius,
                                        dist=dist)
    elif sensors == 'separate':
        dtw = 0.0
        path = []
        for i_sensor in range(sen_data0.shape[0]):
            if radius < 0:
                curr_dist, curr_path = fastdtw.dtw(np.transpose(np.squeeze(sen_data0[i_sensor, :])),
                                                   np.transpose(np.squeeze(sen_data1[i_sensor, :])),
                                                   dist=dist)
            else:
                curr_dist, curr_path = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data0[i_sensor, :])),
                                                       np.transpose(np.squeeze(sen_data1[i_sensor, :])),
                                                       radius=radius,
                                                       dist=dist)
            curr_path = np.array(curr_path)
            dtw += curr_dist
            path.append(curr_path)
    elif sensors == 'three':
        dtw = 0.0
        path = []
        for i_sensor in range(0, sen_data0.shape[0], 3):
            if radius < 0:
                curr_dist, curr_path = fastdtw.dtw(np.transpose(np.squeeze(sen_data0[i_sensor:(i_sensor + 3), :])),
                                                   np.transpose(np.squeeze(sen_data1[i_sensor:(i_sensor + 3), :])),radius=radius,
                                                   dist=dist)
            else:
                curr_dist, curr_path = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data0[i_sensor:(i_sensor + 3), :])),
                                                       np.transpose(np.squeeze(sen_data1[i_sensor:(i_sensor + 3), :])),
                                                       radius=radius,
                                                       dist=dist)
            curr_path = np.array(curr_path)
            dtw += curr_dist
            path.append(curr_path)
    else:
        if radius < 0:
            dtw, path = fastdtw.dtw(np.transpose(np.squeeze(sen_data0[2::3, :])),
                                    np.transpose(np.squeeze(sen_data1[2::3, :])),
                                    dist=dist)
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
        for i_sensor in range(0, sen_data0.shape[0], 3):
            na_dist += np.sum([dist(np.squeeze(sen_data0[i_sensor:(i_sensor+3), i]),
                                    np.squeeze(sen_data1[i_sensor:(i_sensor+3), i])) for i in range(sen_data.shape[-1])])
    else:
        na_dist = np.sum([dist(np.squeeze(sen_data0[2::3, i]),
                               np.squeeze(sen_data1[2::3, i])) for i in range(sen_data.shape[-1])])
    return na_dist


def make_cost_matrix(sen_data0, sen_data1, dist, sensors):
    t0 = sen_data0.shape[-1]
    t1 = sen_data1.shape[-1]
    cost_mat = np.empty((t0, t1))
    for i in range(t0):
        for j in range(t1):
            if sensors == 'all':
                cost_mat[i, j] = dist(np.squeeze(sen_data0[:, i]), np.squeeze(sen_data1[:, j]))
            elif sensors == 'separate':
                cost_mat[i, j] = 0.0
                for i_sensor in range(sen_data0.shape[0]):
                    cost_mat[i, j] += dist(np.squeeze(sen_data0[i_sensor, i]), np.squeeze(sen_data1[i_sensor, j]))
            elif sensors == 'three':
                cost_mat[i, j] = 0.0
                for i_sensor in range(0, sen_data0.shape[0], 3):
                    cost_mat[i, j] += dist(np.squeeze(sen_data0[i_sensor:(i_sensor+3), i]), np.squeeze(sen_data1[i_sensor:(i_sensor+3), j]))
            else:
                cost_mat[i, j] = dist(np.squeeze(sen_data0[2::3, i]), np.squeeze(sen_data1[2::3, j]))
    return cost_mat


def warp_data(sen_data, path, path_ind, sensors):
    if sensors == 'all' or sensors == 'mag':
        warp_sen_data = np.squeeze(sen_data[:, path[:, path_ind]])
    elif sensors == 'separate':
        warp_sen_data = []
        min_time = 100000
        for i_path, sensor_path in enumerate(path):
            time = sensor_path.shape[0]
            if time < min_time:
                min_time = time
            warp_sen_data.append(np.squeeze(sen_data[i_path, sensor_path[:, path_ind]]))
        warp_sen_data = np.concatenate([wdata[None, :min_time] for wdata in warp_sen_data], axis=0)
    else:
        warp_sen_data = []
        min_time = 100000
        for i_path, sensor_path in enumerate(path):
            i_sensor = i_path*3
            time = sensor_path.shape[0]
            if time < min_time:
                min_time = time
            warp_sen_data.append(np.squeeze(sen_data[i_sensor:(i_sensor+3), sensor_path[:, path_ind]]))
        warp_sen_data = np.concatenate([wdata[:, :min_time] for wdata in warp_sen_data], axis=0)
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
    parser.add_argument('--radius', type=int) # if -1 use exact dtw
    parser.add_argument('--num_instances', type=int)
    parser.add_argument('--rep', type=int)
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--sen1', type=int, default=6)
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tmax', type=float, default=0.3)
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

    fname_within = '/home/nrafidi/thesis_figs/{exp}_{sub}_dtw_rep_within_ni{ni}_' \
                   '{sen_type}_r{rad}_rep{rep0}_{tmin}-{tmax}_{sensors}_{dist}.png'.format(exp=exp,
                                                                                           sub=sub,
                                                                                           rad=radius,
                                                                                           ni=num_instances,
                                                                                           sen_type=sen_type,
                                                                                           rep0=rep0,
                                                                                           tmin=tmin,
                                                                                           tmax=tmax,
                                                                                           sensors=sensors,
                                                                                           dist=args.dist)

    fname_without = '/home/nrafidi/thesis_figs/{exp}_{sub}_dtw_rep_without_ni{ni}_' \
                   '{sen_type}_r{rad}_rep{rep0}_{tmin}-{tmax}_{sensors}_{dist}.png'.format(exp=exp,
                                                                                           sub=sub,
                                                                                           rad=radius,
                                                                                           ni=num_instances,
                                                                                           sen_type=sen_type,
                                                                                           rep0=rep0,
                                                                                           tmin=tmin,
                                                                                           tmax=tmax,
                                                                                           sensors=sensors,
                                                                                           dist=args.dist)
    fname_cost = '/home/nrafidi/thesis_figs/{exp}_{sub}_dtw_cost_ni{ni}_' \
                 '{sen_type}_r{rad}_rep{rep0}_{tmin}-{tmax}_{sensors}_{dist}.png'.format(exp=exp,
                                                                                         sub=sub,
                                                                                         rad=radius,
                                                                                         ni=num_instances,
                                                                                         sen_type=sen_type,
                                                                                         rep0=rep0,
                                                                                         tmin=tmin,
                                                                                         tmax=tmax,
                                                                                         sensors=sensors,
                                                                                         dist=args.dist)

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

    cost_mat_within = make_cost_matrix(sen_data[rep0, :, :], sen_data[rep1, :, :], dist, sensors)

    dtw_without, path_without = apply_dtw(sen_data[rep0, :, :], sen_data[rep0 + num_instances, :, :], radius, dist, sensors)

    path_without = np.array(path_without)
    print('Across sentence dtw distance: {}'.format(dtw_without))

    cost_mat_without = make_cost_matrix(sen_data[rep0, :, :], sen_data[rep0 + num_instances, :, :], dist, sensors)

    dist_noalign_within = noalign_dist(sen_data[rep0, :, :], sen_data[rep1, :, :], dist, sensors)

    print('Within sentence no align distance: {}'.format(dist_noalign_within))

    dist_noalign_without = noalign_dist(sen_data[rep0, :, :], sen_data[rep0+num_instances, :, :], dist, sensors)

    print('Across sentence no align distance: {}'.format(dist_noalign_without))

    orig_rep0_data = np.squeeze(sen_data[rep0, :, :])
    orig_rep0_data /= np.max(np.abs(orig_rep0_data))

    warp_rep0_data = warp_data(sen_data[rep0, :, :], path_within, 0, sensors)
    warp_rep0_data /= np.max(np.abs(warp_rep0_data))

    orig_rep1_data = np.squeeze(sen_data[rep1, :, :])
    orig_rep1_data /= np.max(np.abs(orig_rep1_data))

    warp_rep1_data = warp_data(sen_data[rep1, :, :], path_within, 1, sensors)
    warp_rep1_data /= np.max(np.abs(warp_rep1_data))

    max_cost = np.max(np.concatenate([cost_mat_within, cost_mat_without], axis=0))
    min_cost = np.min(np.concatenate([cost_mat_within, cost_mat_without], axis=0))


    time_labels_float = np.linspace(tmin, tmax, 10)
    time_labels_str = ['%.2f' % tl for tl in time_labels_float]


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    h0 = axs[0].imshow(cost_mat_within, interpolation='nearest', aspect='auto', vmin=min_cost, vmax=max_cost)
    axs[0].set_yticks(np.linspace(0, cost_mat_within.shape[0], 10))
    axs[0].set_yticklabels(time_labels_str)
    axs[0].set_xticks(np.linspace(0, cost_mat_within.shape[0], 10))
    axs[0].set_xticklabels(time_labels_str)
    axs[0].set_title('Cost Matrix Within Sentence')
    h1 = axs[1].imshow(cost_mat_without, interpolation='nearest', aspect='auto', vmin=min_cost, vmax=max_cost)
    axs[1].set_yticks(np.linspace(0, cost_mat_without.shape[0], 10))
    axs[1].set_yticklabels(time_labels_str)
    axs[1].set_xticks(np.linspace(0, cost_mat_without.shape[0], 10))
    axs[1].set_xticklabels(time_labels_str)
    axs[1].set_title('Cost Matrix Across Sentence')

    if sensors == 'all' or sensors == 'mag':
        for i in range(path_within.shape[0]):
            axs[0].scatter(path_within[i, 1], path_within[i, 0], c='k', marker='x')
        for i in range(path_without.shape[0]):
            axs[1].scatter(path_without[i, 1], path_without[i, 0], c='k', marker='x')

    fig.suptitle('{sen_type} rep {rep} sen {sen0} vs sen {sen1} ninst {ninst}\n'
                 '{sensors} sensors {dist}'.format(sen_type=sen_type,
                                                   rep=rep0,
                                                   sen0=sen0,
                                                   sen1=sen1,
                                                   ninst=num_instances,
                                                   sensors=sensors,
                                                   dist=args.dist),
                 fontsize=18)
    plt.subplots_adjust(top=0.8)
    plt.savefig(fname_cost, bbox_inches='tight')

    fig, axs = plt.subplots(2, 2)
    h00 = axs[0][0].imshow(orig_rep0_data, interpolation='nearest', aspect='auto')
    axs[0][0].set_title('Original Sen {sen0} Rep {rep0}'.format(sen0=sen0,
                                                                rep0=rep0))
    axs[0][0].set_xticks(np.linspace(0, orig_rep0_data.shape[-1], 10))
    axs[0][0].set_xticklabels(time_labels_str)
    h01 = axs[0][1].imshow(warp_rep0_data, interpolation='nearest', aspect='auto')
    axs[0][1].set_title('Warped Sen {sen0} Rep {rep0}'.format(sen0=sen0,
                                                              rep0=rep0))
    axs[0][1].set_xticks(np.linspace(0, warp_rep0_data.shape[-1], 10))
    axs[0][1].set_xticklabels(time_labels_str)
    h10 = axs[1][0].imshow(orig_rep1_data, interpolation='nearest', aspect='auto')
    axs[1][0].set_title('Original Sen {sen0} Rep {rep1}'.format(sen0=sen0,
                                                                rep1=rep1))
    axs[1][0].set_xticks(np.linspace(0, orig_rep1_data.shape[-1], 10))
    axs[1][0].set_xticklabels(time_labels_str)
    h11 = axs[1][1].imshow(warp_rep1_data, interpolation='nearest', aspect='auto')
    axs[1][1].set_title('Warped Sen {sen0} Rep {rep1}'.format(sen0=sen0,
                                                              rep1=rep1))
    axs[1][1].set_xticks(np.linspace(0, warp_rep1_data.shape[-1], 10))
    axs[1][1].set_xticklabels(time_labels_str)
    fig.suptitle('Within Sentence\nDTW: {} No Align: {}'.format(dtw_within, dist_noalign_within), fontsize=18)
    plt.subplots_adjust(top=0.85)
    plt.savefig(fname_within,
                bbox_inches='tight')

    orig_sen0_data = np.squeeze(sen_data[rep0, :, :])
    orig_sen0_data /= np.max(np.abs(orig_sen0_data))

    warp_sen0_data = warp_data(sen_data[rep0, :, :], path_without, 0, sensors)
    warp_sen0_data /= np.max(np.abs(warp_sen0_data))

    orig_sen1_data = np.squeeze(sen_data[num_instances, :, :])
    orig_sen1_data /= np.max(np.abs(orig_sen1_data))

    warp_sen1_data = warp_data(sen_data[rep0+num_instances, :, :], path_within, 1, sensors)
    warp_sen1_data /= np.max(np.abs(warp_sen1_data))

    # print(np.sum(np.equal(warp_rep0_data, orig_rep0_data)))
    fig, axs = plt.subplots(2, 2)
    h00 = axs[0][0].imshow(orig_sen0_data, interpolation='nearest', aspect='auto')
    axs[0][0].set_title('Original Sen {sen0} Rep {rep0}'.format(sen0=sen0,
                                                                rep0=rep0))
    axs[0][0].set_xticks(np.linspace(0, orig_sen0_data.shape[-1], 10))
    axs[0][0].set_xticklabels(time_labels_str)
    h01 = axs[0][1].imshow(warp_sen0_data, interpolation='nearest', aspect='auto')
    axs[0][1].set_title('Warped Sen {sen0} Rep {rep0}'.format(sen0=sen0,
                                                              rep0=rep0))
    axs[0][1].set_xticks(np.linspace(0, warp_sen0_data.shape[-1], 10))
    axs[0][1].set_xticklabels(time_labels_str)
    h10 = axs[1][0].imshow(orig_sen1_data, interpolation='nearest', aspect='auto')
    axs[1][0].set_title('Original Sen {sen1} Rep {rep0}'.format(sen1=sen1,
                                                                rep0=rep0))
    axs[1][0].set_xticks(np.linspace(0, orig_sen1_data.shape[-1], 10))
    axs[1][0].set_xticklabels(time_labels_str)
    h11 = axs[1][1].imshow(warp_sen1_data, interpolation='nearest', aspect='auto')
    axs[1][1].set_title('Warped Sen {sen1} Rep {rep0}'.format(sen1=sen1,
                                                              rep0=rep0))
    axs[1][1].set_xticks(np.linspace(0, warp_sen1_data.shape[-1], 10))
    axs[1][1].set_xticklabels(time_labels_str)
    fig.suptitle('Across Sentence\nDTW: {} No Align: {}'.format(dtw_without, dist_noalign_without), fontsize=18)

    plt.subplots_adjust(top=0.85)
    plt.savefig(fname_without,
                bbox_inches='tight')
    plt.show()