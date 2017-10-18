import argparse
import load_data
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.spatial.distance.pdist as pdist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    args = parser.parse_args()

    word = 'firstNoun'

    evokeds, labels, time = load_data.load_raw(args.subject, word, 'active',
                                               experiment=args.experiment, proc=args.proc)
    act_data, labels_act = load_data.avg_data(evokeds, labels, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)

    evokeds, labels, _ = load_data.load_raw(args.subject, word, 'passive',
                                               experiment=args.experiment, proc=args.proc)
    pass_data, labels_pass = load_data.avg_data(evokeds, labels, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)

    pass_data = pass_data[:, :, time.shape[0]]

    total_data = np.concatenate((act_data, pass_data), axis=0)
    total_data = np.reshape(total_data, (total_data.shape[0], -1))

    rdm = pdist(total_data)

    fig, ax = plt.subplots()
    h = ax.imshow(rdm, interpolation='nearest')
    plt.colorbar(h)
    plt.show()