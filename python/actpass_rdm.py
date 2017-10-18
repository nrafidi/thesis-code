import argparse
import load_data
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform



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

    evokeds, labels, time, sen_ids = load_data.load_raw(args.subject, word, 'active',
                                               experiment=args.experiment, proc=args.proc, tmin=0.0, tmax=2.0)
    act_data, labels_act, sen_ids_act = load_data.avg_data(evokeds, labels, sentence_ids_raw=sen_ids, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)
    labels_act = np.array(labels_act)
    label_sort_inds = np.argsort(sen_ids_act)

    act_data = act_data[label_sort_inds, :, :]

    print(act_data.shape)
    evokeds, labels, _, sen_ids = load_data.load_raw(args.subject, word, 'passive',
                                               experiment=args.experiment, proc=args.proc, tmin=0.0, tmax=2.0)
    pass_data, labels_pass, sen_ids_pass = load_data.avg_data(evokeds, labels, sentence_ids_raw=sen_ids, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)
    labels_pass = np.array(labels_pass)
    pass_data = pass_data[:, :, :time.shape[0]]
    label_sort_inds = np.argsort(sen_ids_pass)
    pass_data = pass_data[label_sort_inds, :, :]
    print(pass_data.shape)

    print(labels_pass[label_sort_inds])

    total_data = np.concatenate((act_data, pass_data), axis=0)
    total_data = np.reshape(total_data, (total_data.shape[0], -1))

    rdm = squareform(pdist(total_data))

    fig, ax = plt.subplots()
    h = ax.imshow(rdm, interpolation='nearest')
    plt.colorbar(h)
    plt.show()