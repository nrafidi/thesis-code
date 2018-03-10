import argparse
import numpy as np
from syntax_vs_semantics import load_data
from scipy.spatial.distance import euclidean, cosine
import fastdtw


RESULT_FNAME = '/share/volume0/nrafidi/DTW/EOS_dtw_sensor{i_sensor}_score_{exp}_{sub}_sen{sen0}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--dist', choices=['euclidean', 'cosine'])
    parser.add_argument('--radius', type=int)
    parser.add_argument('--num_instances', type=int)
    parser.add_argument('--tmin', type=float, default=0.0)
    parser.add_argument('--tmax', type=float, default=1.0)
    parser.add_argument('--sensor', type=int)
    parser.add_argument('--sen0', type=int, default=0)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)

    args = parser.parse_args()

    exp = args.experiment
    sub = args.subject
    proc = args.proc
    radius = args.radius
    num_instances = args.num_instances
    sen0 = args.sen0
    tmin = args.tmin
    tmax = args.tmax
    i_sensor=args.sensor

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
    data = data*1e12
    sen_ints = np.array(sen_ints)

    sen0_data = data[sen_ints == sen0, ...]
    other_sens = range(sen0, np.max(sen_ints)+1)

    dtw_part = np.empty((len(other_sens), num_instances, num_instances))
    for i_sen1, sen1 in enumerate(other_sens):
        sen1_data = data[sen_ints == sen1, ...]
        for i in range(num_instances):
            for j in range(num_instances):
                dtw_part[i_sen1, i, j], _ = fastdtw.fastdtw(np.transpose(np.squeeze(sen0_data[i, i_sensor, :])),
                                                            np.transpose(np.squeeze(sen1_data[j, i_sensor, :])),
                                                            radius=radius,
                                                            dist=dist)

    np.savez(RESULT_FNAME.format(exp=exp, sub=sub, sen0=sen0, radius=radius, dist=args.dist,
                                 ni=num_instances, tmin=tmin, tmax=tmax, i_sensor=i_sensor),
             dtw_part = dtw_part)
