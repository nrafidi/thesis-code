import argparse
import numpy as np
from syntax_vs_semantics import load_data
from scipy.spatial.distance import euclidean, cosine
import fastdtw
import os


RESULT_FNAME = '/share/volume0/nrafidi/DTW/EOS_{metric}_sensor{i_sensor}_score_{exp}_{sub}_sen{sen0}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'


def str_to_bool(str_bool):
    if str_bool == 'False':
        return False
    else:
        return True


def total_dist(series1, series2, do_transpose=False, dist=euclidean):
    num_time_points = series1.shape[-1]

    tot_dist = 0.0
    for i_time in range(num_time_points):
        if do_transpose:
            dat1 = series1[i_time]
            dat2 = series2[i_time]
        else:
            dat1 = series1[:, i_time]
            dat2 = series2[:, i_time]
        tot_dist += dist(dat1, dat2)
    return tot_dist



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--metric', default='dtw', choices=['dtw', 'total'])
    parser.add_argument('--dist', choices=['euclidean', 'cosine'])
    parser.add_argument('--radius', type=int)
    parser.add_argument('--num_instances', type=int)
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tim_len', type=float, default=0.1)
    parser.add_argument('--sensor', type=int)
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--force', default='False', choices=['True', 'False'])

    args = parser.parse_args()

    exp = args.experiment
    metric = args.metric
    sub = args.subject
    proc = args.proc
    radius = args.radius
    num_instances = args.num_instances
    sen0 = args.sen0
    tmin = args.tmin
    tmax = tmin + args.time_len
    i_sensor=args.sensor

    if args.dist == 'euclidean':
        dist=euclidean
    else:
        dist=cosine

    result_fname = RESULT_FNAME.format(metric=metric, exp=exp, sub=sub, sen0=sen0, radius=radius, dist=args.dist,
                                 ni=num_instances, tmin=tmin, tmax=tmax, i_sensor=i_sensor)

    if os.path.isfile(result_fname) and not str_to_bool(args.force):
        print('Job already completed. Skipping Job.')
        print(result_fname)
    else:
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
        data = data*1e12
        sen_ints = np.array(sen_ints)

        if i_sensor < 0:
            do_transpose=False
            i_sensor = np.ones((data.shape[1],), dtype=bool)
        else:
            do_transpose=True

        sen0_data = data[sen_ints == sen0, ...]
        other_sens = range(sen0, np.max(sen_ints)+1)

        dtw_part = np.empty((len(other_sens), num_instances, num_instances))
        for i_sen1, sen1 in enumerate(other_sens):
            sen1_data = data[sen_ints == sen1, ...]
            for i in range(num_instances):
                for j in range(num_instances):
                    if do_transpose:
                        series1 = np.transpose(np.squeeze(sen0_data[i, i_sensor, :]))
                        series2 = np.transpose(np.squeeze(sen1_data[j, i_sensor, :]))
                    else:
                        series1 = np.squeeze(sen0_data[i, i_sensor, :])
                        series2 = np.squeeze(sen1_data[j, i_sensor, :])
                    if metric == 'dtw':
                        dtw_part[i_sen1, i, j], _ = fastdtw.fastdtw(series1,
                                                                    series2,
                                                                    radius=radius,
                                                                    dist=dist)
                    else:
                        dtw_part[i_sen1, i, j] = total_dist(series1, series2, do_transpose=do_transpose,
                                                            dist=dist)

        np.savez(result_fname,
                 dtw_part = dtw_part)
