import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO_EOS
import tgm_loso_acc_eos
import string


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--win_len', type=int, default=25)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=5)
    parser.add_argument('--avgTime', default='T')
    parser.add_argument('--avgTest', default='T')
    args = parser.parse_args()

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    sen_type = 'pooled'
    word_list = ['verb', 'agent', 'patient', 'voice']

    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002

    num_sub = float(len(run_TGM_LOSO_EOS.VALID_SUBS[args.experiment]))
    frac_thresh = (0.5*num_sub + 1.0)/num_sub

    sen_fig, ax = plt.subplots(figsize=(10, 10))
    acc_diags = []
    frac_diags = []
    time = []
    win_starts = []
    for word in word_list:
        intersection, acc_all, word_time, word_win_starts, eos_max = tgm_loso_acc_eos.intersect_accs(args.experiment,
                                                                                                    sen_type,
                                                                                                    word,
                                                                                                    win_len=args.win_len,
                                                                                                    overlap=args.overlap,
                                                                                                    adj=args.adj,
                                                                                                    num_instances=args.num_instances,
                                                                                                    avgTime=args.avgTime,
                                                                                                    avgTest=args.avgTest)

        frac_diags.append(np.diag(intersection).astype('float')/float(acc_all.shape[0]))
        acc_diags.append(np.diag(np.mean(acc_all, axis=0)))
        if word == 'verb':
            time = word_time
            win_starts = word_win_starts


    max_line = 0.3 * 2 * time_step
    colors = ['r', 'g', 'b', 'm', 'k']

    for i_word, word in enumerate(word_list):
        color = colors[i_word]
        acc = acc_diags[i_word]
        frac = frac_diags[i_word]

        above_thresh = frac > frac_thresh
        ax.plot(acc, label='{word} accuracy'.format(word=word), color=color)
        for i_pt, pt in enumerate(above_thresh):
            if pt:
                ax.scatter(i_pt, acc[i_pt] + 0.05, color=color, marker='*')

    ax.set_xticks(range(0, len(time[win_starts]), time_step))
    label_time = time[win_starts]
    label_time = label_time[::time_step]
    label_time[np.abs(label_time) < 1e-15] = 0.0
    ax.axhline(y=0.25, color='k', linestyle='dashed', label='chance accuracy, words')
    ax.axhline(y=0.5, color='k', linestyle='dashdot', label='chance accuracy, voice')
    ax.set_xticklabels(label_time)
    ax.axvline(x=max_line, color='k')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Time Relative to First Noun Onset (s)')
    ax.set_ylim([0.0, 0.9])
    ax.legend(loc=1)

    sen_fig.suptitle('Mean Accuracy over Subjects', fontsize=18)
    sen_fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_{sen_type}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            exp=args.experiment, sen_type=sen_type, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    plt.show()


