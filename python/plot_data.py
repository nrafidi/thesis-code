import argparse
import load_data
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['r', 'b', 'g', 'k', 'c', 'm']

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

    evokeds, labels, time = load_data.load_raw(args.subject, args.word, args.sen_type,
                                               experiment=args.experiment, proc=args.proc)

    avg_data, labels_avg = load_data.avg_data(evokeds, labels, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)
    print(avg_data.shape)
    print(labels_avg)
    print(type(labels_avg))

    # fig, ax = plt.subplots()
