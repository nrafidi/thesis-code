import argparse
import load_data
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

COLORS = ['r', 'b', 'g', 'k', 'c', 'm']

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


def baseline_correct(evokeds, time):
    baseline = np.mean(evokeds[:, :, :, time < 0], axis=3)
    for t in range(evokeds.shape[3]):
        evokeds[:, :, :, t] -= baseline
    return evokeds


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

    evokeds, labels, time, _ = load_data.load_raw(args.subject, args.word, args.sen_type,
                                                  experiment=args.experiment, proc=args.proc)
    evokeds = baseline_correct(evokeds, time)

    avg_data, labels_avg, _ = load_data.avg_data(evokeds, labels, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)

    sorted_inds, sorted_reg = sort_sensors()
    avg_data = avg_data[:, sorted_inds, :]
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    num_time = time.size

    for i in range(avg_data.shape[0]):
        fig, ax = plt.subplots()
        h = ax.imshow(np.squeeze(avg_data[i, :, :]), interpolation='nearest', aspect='auto', vmin=-0.4e-11, vmax=0.4e-11)
        ax.set_yticks(yticks_sens)
        ax.set_yticklabels(uni_reg)
        ax.set_ylabel('Sensors')
        ax.set_xticks(range(0, num_time, 250))
        ax.set_xticklabels(time[::250])
        ax.set_xlabel('Time')
        plt.colorbar(h)
    plt.show()

    # uni_labels = [lab for lab in set(labels_avg)]
    #
    # sorted_inds, sorted_reg = sort_sensors()
    # avg_data = avg_data[:, sorted_inds, :]
    #
    # first_r_p = sorted_reg.index('R_Parietal') + 1
    #
    # sensor_to_plot = np.squeeze(avg_data[:, first_r_p, :])
    #
    # num_sentences = avg_data.shape[0]
    #
    # avg_over_labels = np.empty((len(uni_labels), sensor_to_plot.shape[1]))
    # fig, ax = plt.subplots()
    # for lab in uni_labels:
    #     inds =[i for i, x in enumerate(labels_avg) if x == lab]
    #     label_ind = uni_labels.index(lab)
    #     avg_over_labels[label_ind, :] = np.mean(sensor_to_plot[inds, :], axis=0)
    #     ax.plot(time[::25], avg_over_labels[label_ind, ::25], COLORS[label_ind])
    # plt.show()

    # fig, ax = plt.subplots()
    # for i_sen in range(num_sentences):
    #     label_ind = uni_labels.index(labels_avg[i_sen])
    #     ax.plot(time[::25], sensor_to_plot[i_sen, ::25], COLORS[label_ind])
    # plt.show()
