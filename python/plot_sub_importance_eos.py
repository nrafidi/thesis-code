import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
from math import ceil
import run_TGM_LOSO_EOS
from mpl_toolkits.axes_grid1 import AxesGrid
import string
import run_coef_TGM_EOS_multisub
from mne.viz import plot_topomap
from syntax_vs_semantics import sensor_metadata

PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice',
                   'propid': 'Proposition ID',
                   'senlen': 'Sentence Length'}


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'
CHANCE = {'krns2':{'pooled': {'noun1': 0.125,
                             'verb': 0.25,
                             'voice': 0.5,
                              'agent': 0.25,
                              'patient': 0.25,
                              'propid': 1.0/16.0},
                  'active': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                             'agent': 0.25,
                             'patient': 0.25,
                             'propid': 1.0 / 8.0
                             },
                  'passive': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                              'agent': 0.25,
                              'patient': 0.25,
                              'propid': 1.0 / 8.0
                              }
                    },
          'PassAct3': {'pooled': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                                  'agent': 0.25,
                                  'patient': 0.25,
                              'propid': 1.0/8.0,
                                  'senlen': 0.5},
                  'active': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                                  'agent': 0.25,
                                  'patient': 0.25,
                              'propid': 1.0/8.0,},
                  'passive': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                                  'agent': 0.25,
                                  'patient': 0.25,
                              'propid': 1.0/8.0,}
                    }}

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'
MULTI_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{rank_str}{mode}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_TGM_LOSO_EOS.VALID_SEN_TYPE)
    # parser.add_argument('--word', choices = ['noun1', 'verb', 'voice', 'agent', 'patient'])
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='T')
    parser.add_argument('--avgTest', default='T')
    parser.add_argument('--time_to_plot', type=float, default=1.44)
    args = parser.parse_args()

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    sen_type = args.sen_type
    word_list = ['agent', 'patient', 'verb']
    if args.experiment == 'krns2':
        if sen_type == 'pooled':
            word_list.extend(['noun1', 'voice', 'propid'])
    else:
        if sen_type == 'pooled':
            word_list.extend(['senlen', 'noun1', 'voice', 'propid'])
    if sen_type == 'pooled':
        n_rows=2
    else:
        n_rows=1

    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step

    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']

    combo_fig, ax = plt.subplots()

    ind = np.arange(len(run_coef_TGM_EOS_multisub.VALID_SUBS[args.experiment]))  # the x locations for the groups
    width = 0.8 / len(word_list)  # the width of the bars

    subject_scores = np.empty((len(run_coef_TGM_EOS_multisub.VALID_SUBS[args.experiment]),
                               len(word_list)))
    accs = np.empty((len(word_list),))
    for i_word, word in enumerate(word_list):
        top_dir = TOP_DIR.format(exp=args.experiment)
        fname = run_coef_TGM_EOS_multisub.SAVE_FILE.format(dir=top_dir,
                                                       sen_type=sen_type,
                                                       word=word,
                                                       win_len=args.win_len,
                                                       ov=args.overlap,
                                                       alg=args.alg,
                                                       adj=args.adj,
                                                       avgTm=args.avgTime,
                                                       inst=args.num_instances)

        result = np.load(fname + '.npz')
        maps = result['haufe_maps']
        win_starts = result['win_starts']
        time_win = result['time'][win_starts]

        rank_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                           sen_type=sen_type,
                                           word=word,
                                           win_len=args.win_len,
                                           ov=args.overlap,
                                           perm='F',
                                           alg=args.alg,
                                           adj=args.adj,
                                           avgTm=args.avgTime,
                                           avgTst=args.avgTest,
                                           inst=args.num_instances,
                                           rsP=1,
                                           rank_str='rank',
                                           mode='acc')

        rank_result = np.load(rank_file + '.npz')
        multi_fold_acc = rank_result['tgm_rank']
        mean_acc = np.diag(np.mean(multi_fold_acc, axis=0))
        post_onset = time_win >= 0.0

        maps = maps[post_onset]
        mean_acc = mean_acc[post_onset]
        time_win = time_win[post_onset]

        i_plot = np.where(np.logical_and(time_win >= args.time_to_plot, time_win < args.time_to_plot + 0.1))[0][0]
        # print(i_plot)
        max_acc = mean_acc[i_plot]
        map = maps[i_plot]
        i_max_acc = np.argmax(mean_acc)
        max_time = time_win[i_max_acc]
        print(max_time)

        accs[i_word] = max_acc
        # time_to_plot = time_win[max_acc]
        #
        # map = maps[max_acc]
        class_map = np.mean(map, axis=0)
        sub_map = np.zeros((306,))
        for i_sub in range(len(run_coef_TGM_EOS_multisub.VALID_SUBS[args.experiment])):
            start_ind = i_sub * 306
            end_ind = start_ind + 306
            sub_map = class_map[start_ind:end_ind]
            subject_scores[i_sub, i_word] = np.sum(np.abs(sub_map))

        ax.bar(ind + i_word * width, subject_scores[:, i_word], width,
               color=colors[i_word], label='%s: %.2f' % (PLOT_TITLE_WORD[word], max_acc))

    ax.legend(fontsize=legendfontsize)
    ax.set_ylabel('Importance Score', fontsize=axislabelsize)
    ax.set_xlabel('Subject', fontsize=axislabelsize)
    ax.set_xticks(ind + 0.4)
    ax.set_xticklabels(run_coef_TGM_EOS_multisub.VALID_SUBS[args.experiment])
    ax.tick_params(fontsize=ticklabelsize)
    combo_fig.suptitle('Subject Importance Scores at %.2f s Post-Onset' % args.time_to_plot,
        fontsize=suptitlesize)
    combo_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_eos_sub_import_{tplot}_multisub_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=args.experiment, sen_type=sen_type, word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=args.win_len,
                overlap=args.overlap,
                num_instances=args.num_instances,
                id=args.sensor_type_id,
                tplot=args.time_to_plot
            ), bbox_inches='tight')
    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_sub_import_{tplot}_multisub_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=sen_type, word='all', alg=args.alg, avgTime=args.avgTime,
            avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances,
            id=args.sensor_type_id,
            tplot=args.time_to_plot
        ), bbox_inches='tight')
    plt.show()