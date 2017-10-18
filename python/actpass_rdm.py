import argparse
import load_data
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


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
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    args = parser.parse_args()

    word = 'secondNoun'
    sorted_inds, sorted_reg = sort_sensors()

    evokeds, labels, time, sen_ids = load_data.load_raw(args.subject, word, 'active',
                                               experiment=args.experiment, proc=args.proc, tmin=0.0, tmax=0.5)
    act_data, labels_act, sen_ids_act = load_data.avg_data(evokeds, labels, sentence_ids_raw=sen_ids, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)
    labels_act = np.array(labels_act)
    label_sort_inds = np.argsort(sen_ids_act)

    act_data = act_data[label_sort_inds, :, :]
    act_data = act_data[:, sorted_inds, :]

    print(act_data.shape)
    evokeds, labels, _, sen_ids = load_data.load_raw(args.subject, word, 'passive',
                                               experiment=args.experiment, proc=args.proc, tmin=0.0, tmax=0.5)
    pass_data, labels_pass, sen_ids_pass = load_data.avg_data(evokeds, labels, sentence_ids_raw=sen_ids, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)
    labels_pass = np.array(labels_pass)
    pass_data = pass_data[:, :, :time.shape[0]]
    label_sort_inds = np.argsort(sen_ids_pass)
    pass_data = pass_data[label_sort_inds, :, :]
    pass_data = pass_data[:, sorted_inds, :]
    print(pass_data.shape)

    print(labels_pass[label_sort_inds])

    total_data = np.concatenate((act_data, pass_data), axis=0)

    for reg in set(sorted_reg):
        locs = [i for i, x in enumerate(sorted_reg) if x == reg]
        reshaped_data = np.reshape(total_data[:, locs, :], (total_data.shape[0], -1))

        rdm = squareform(pdist(reshaped_data))

        fig, ax = plt.subplots()
        h = ax.imshow(rdm, interpolation='nearest', vmin=1e-10, vmax=5e-10)
        ax.set_title(reg)
        plt.colorbar(h)
    plt.show()