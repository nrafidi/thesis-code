import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import string
import tgm_loso_acc_eos
from mpl_toolkits.axes_grid1 import AxesGrid
import run_TGM_LOSO_EOS

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice',
                   'propid': 'Proposition ID',
                   'senlen': 'Sentence Length'}

PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--num_sub', type=int)
    parser.add_argument('--win_len', type=int, default=25)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='F')
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

    win_len = args.win_len
    num_instances = args.num_instances
    experiment = args.experiment

    time_adjust = win_len * 0.002

    num_sub_to_plot = args.num_sub

    ticklabelsize = 14
    legendfontsize = 14
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20
    time_step = int(250 / args.overlap)

    max_line = 0.3 * 2 * time_step
    colors = ['b', 'm', 'g', 'k', 'r', 'c']

    chance = {'krns2': {'noun1': 0.125,
                  'verb': 0.25,
                  'agent': 0.25,
                  'patient': 0.25,
                  'voice': 0.5,
                  'propid': 1.0 / 16.0},
              'PassAct3': {'noun1': 0.25,
                  'verb': 0.25,
                  'agent': 0.25,
                  'patient': 0.25,
                  'voice': 0.5,
                  'propid': 1.0 / 8.0,
                  'senlen': 0.5}}

    word_list = ['verb', 'agent', 'patient']

    sen_type_list = ['active', 'passive', 'pooled']
    for i_sen_type, sen_type in enumerate(sen_type_list):
        if sen_type == 'pooled':
            if num_instances > 1:
                word_list.append('propid')
            word_list.append('voice')
            if experiment == 'krns2':
                word_list.append('noun1')
            else:
                word_list.append('senlen')

        time = []
        win_starts = []
        sub_word_diags = []
        for i_word, word in enumerate(word_list):
            intersection, acc_all, word_time, word_win_starts, eos_max = tgm_loso_acc_eos.intersect_accs(args.experiment,
                                                                                                        sen_type,
                                                                                                        word,
                                                                                                        win_len=win_len,
                                                                                                        overlap=args.overlap,
                                                                                                        alg=args.alg,
                                                                                                        adj=args.adj,
                                                                                                        num_instances=num_instances,
                                                                                                        avgTime=args.avgTime,
                                                                                                        avgTest=args.avgTest)

            num_sub = acc_all.shape[0]
            sub_diags = np.concatenate([np.diag(acc_all[i, :, :])[None, :] for i in range(num_sub)], axis=0)
            sub_word_diags.append(sub_diags[None, :] - chance[experiment][word])
            if i_word == 0:
                time = word_time
                win_starts = word_win_starts

        sub_word_diags = np.concatenate(sub_word_diags, axis=0)

        sub_scores = np.max(sub_word_diags, axis=2)

        meow = np.max(sub_scores, axis=0)

        sub_scores /= meow[None, :]
        sub_scores = np.sum(sub_scores, axis=0)

        print(sub_scores)

        sorted_subs = np.argsort(sub_scores)

        worst_subs = sorted_subs[:num_sub_to_plot]
        best_subs = sorted_subs[(num_sub - 1):(num_sub - 1 - num_sub_to_plot):-1]

        num_time = len(win_starts)
        label_time = time[win_starts]
        label_time = label_time[::time_step]
        label_time[np.abs(label_time) < 1e-15] = 0.0

        curr_fig = plt.figure(figsize=(16, 8 * num_sub_to_plot))
        curr_axs = AxesGrid(curr_fig, 111, nrows_ncols=(num_sub_to_plot, 2),
                            axes_pad=0.7, share_all=True, direction='row', aspect=False)
        i_ax = 0
        for j_sub in range(num_sub_to_plot):
            for i_word, word in enumerate(word_list):
                color = colors[i_word]

                good_sub = sub_word_diags[i_word, best_subs[j_sub]]
                bad_sub = sub_word_diags[i_word, worst_subs[j_sub]]

                curr_axs[i_ax].plot(good_sub, label='{word}'.format(word=PLOT_TITLE_WORD[word]), color=color)
                curr_axs[i_ax + 1].plot(bad_sub, label='{word}'.format(word=PLOT_TITLE_WORD[word]), color=color)
                curr_axs[i_ax].axhline(y=chance[experiment][word], color='k', linestyle='dashed')
                curr_axs[i_ax + 1].axhline(y=chance[experiment][word], color='k', linestyle='dashed')


            for j_ax in range(2):
                curr_axs[i_ax + j_ax].set_xticks(range(0, len(time[win_starts]), time_step))
                curr_axs[i_ax + j_ax].set_xticklabels(label_time)
                curr_axs[i_ax + j_ax].axvline(x=max_line, color='k')

                curr_axs[i_ax + j_ax].set_ylim([0.0, 1.0])
                curr_axs[i_ax + j_ax].set_xlim([0, len(time[win_starts]) + 0.8*time_step])
                curr_axs[i_ax + j_ax].tick_params(labelsize=ticklabelsize)

            good_sub_name = run_TGM_LOSO_EOS.VALID_SUBS[args.experiment][best_subs[j_sub]]
            bad_sub_name = run_TGM_LOSO_EOS.VALID_SUBS[args.experiment][worst_subs[j_sub]]

            curr_axs[i_ax].set_title('Good Subject: {}'.format(good_sub_name), fontsize=axistitlesize)
            curr_axs[i_ax + 1].set_title('Bad Subject: {}'.format(bad_sub_name), fontsize=axistitlesize)
            curr_axs[i_ax + 1].legend(bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0., ncol=1, fontsize=legendfontsize)

            i_ax += 2

        curr_fig.suptitle('Best/Worst {num} Subjects\n{sen_type} {experiment}'.format(num=num_sub_to_plot,
                                                                                      sen_type=sen_type,
                                                                                      experiment=args.experiment),
                          fontsize=suptitlesize)
        curr_fig.text(0.52, 0.08, 'Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
        curr_fig.text(0.04, 0.48, 'Accuracy', va='center',
                      rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
        curr_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_top-bot{num}_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=args.experiment, sen_type=sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=win_len,
                overlap=args.overlap,
                num_instances=num_instances,
                num=num_sub_to_plot
            ), bbox_inches='tight')

    plt.show()


