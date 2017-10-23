import argparse
import load_data
import numpy as np
import scipy.io as sio

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'
SAVE_FNAME = '/share/volume0/newmeg/{}/tom/{}_{}_{}_{}.mat'


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--sen_type', default='active')
    parser.add_argument('--word', default='firstNoun')
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    args = parser.parse_args()

    evokeds, labels, time, sentence_ids = load_data.load_raw(args.subject, args.word, args.sen_type,
                                                             experiment=args.experiment, proc=args.proc)

    fname_raw = SAVE_FNAME.format(args.experiment, args.subject, args.sen_type, args.word, 'raw')

    sio.savemat(fname_raw, mdict={'evokeds': evokeds, 'labels': labels, 'time': time, 'sentence_ids': sentence_ids})

    avg_data, labels_avg = load_data.avg_data(evokeds, labels, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)

    fname_avg = SAVE_FNAME.format(args.experiment, args.subject, args.sen_type, args.word, 'avg' + str(args.reps_to_use))