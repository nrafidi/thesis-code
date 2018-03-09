import argparse
import numpy as np
from syntax_vs_semantics import load_data
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import kendalltau
import fastdtw


RESULT_FNAME = '/share/volume0/nrafidi/DTW/EOS_dtw_sensor_score_{sen0}vs{sen1}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'


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
    tmin = args.tmin
    tmax = args.tmax

    if args.dist == 'euclidean':
        dist=euclidean
    else:
        dist=cosine

    data, labels, sen_ints, time, sensor_regions = load_data.load_sentence_data_v2(subject=sub,
                                                                                   align_to='last',
                                                                                   voice=['active', 'passive'],
                                                                                   experiment=exp,
                                                                                   proc=proc,
                                                                                   num_instances=num_instances,
                                                                                   reps_filter=None,
                                                                                   sensor_type=None,
                                                                                   is_region_sorted=False,
                                                                                   tmin=tmin,
                                                                                   tmax=tmax)

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

    print(sen_data.shape)
    num_sensors = sen_data.shape[1]

    dtw_mat = np.empty((num_sensors, num_sen, num_sen))
    score_mat = np.empty((num_sensors,))
    for i_sensor in range(sen_data.shape[0]):
        for i in range(num_sen):
            for j in range(i, num_sen):
                dtw_mat[i_sensor, i, j], _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen_data[i, i_sensor, :])),
                                                           np.transpose(np.squeeze(sen_data[j, i_sensor, :])),
                                                           radius=radius,
                                                           dist=dist)
                dtw_mat[i_sensor, j, i] = dtw_mat[i_sensor, i, j]
        score_mat[i_sensor], _ = ktau_rdms(comp_mat, np.squeeze(dtw_mat)[i_sensor, ...])

    np.savez(RESULT_FNAME.format(sen0=sen0, sen1=sen1, radius=radius, dist=args.dist,
                                 ni=num_instances, tmin=tmin, tmax=tmax),
             scores=score_mat, dtw_mat = dtw_mat)
