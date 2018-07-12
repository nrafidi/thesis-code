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
from rank_from_pred import rank_from_pred, rank_from_pred_bind

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
                   'senlen': 'Sentence Length',
                   'bind': 'Argument Binding'}


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
MULTI_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub-subcv_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'


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
    parser.add_argument('--exc', action='store_true')
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

    if args.exc:
        exc_str = '_exc'
    else:
        exc_str = ''

    sen_type = args.sen_type
    word_list = ['agent', 'patient', 'verb']
    if args.experiment == 'krns2':
        if sen_type == 'pooled':
            word_list.extend(['noun1', 'voice', 'propid'])
    else:
        if sen_type == 'pooled':
            word_list.extend(['bind', 'senlen', 'noun1', 'voice', 'propid'])
    if sen_type == 'pooled':
        n_rows=2
    else:
        n_rows=1
    num_plots = int(ceil(float(len(word_list))/float(n_rows)))
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(num_plots*6, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(n_rows, num_plots),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.2, share_all=True)

    for i_word, word in enumerate(word_list):
        top_dir = TOP_DIR.format(exp=args.experiment)
        if word == 'bind':
            word_to_load = 'propid'
            rank_fn = rank_from_pred_bind
        else:
            word_to_load = word
            rank_fn = rank_from_pred
        multi_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                            sen_type=sen_type,
                                            word=word_to_load,
                                            win_len=args.win_len,
                                            exc=exc_str,
                                            ov=args.overlap,
                                            perm='F',
                                            alg=args.alg,
                                            adj=args.adj,
                                            avgTm=args.avgTime,
                                            avgTst=args.avgTest,
                                            inst=args.num_instances,
                                            rsP=1,
                                            mode='acc')
        rank_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                           sen_type=sen_type,
                                           word=word,
                                           win_len=args.win_len,
                                           exc=exc_str,
                                           ov=args.overlap,
                                           perm='F',
                                           alg=args.alg,
                                           adj=args.adj,
                                           avgTm=args.avgTime,
                                           avgTst=args.avgTest,
                                           inst=args.num_instances,
                                           rsP=1,
                                           mode='rankacc')
        # rerun = True
        result = np.load(multi_file + '.npz')
        if os.path.isfile(rank_file + '.npz'): # and not rerun:
            rank_result = np.load(rank_file + '.npz')
            multi_fold_acc = rank_result['tgm_rank']
        else:
            tgm_pred = result['tgm_pred']
            print(tgm_pred.shape)
            print(word)
            assert len(tgm_pred.shape) == 4
            num_sub = tgm_pred.shape[0]
            l_ints = result['l_ints']
            print(l_ints.shape)
            cv_membership_all = result['cv_membership']
            print(len(cv_membership_all))
            multi_fold_acc = []
            for i_sub in range(num_sub):
                cv_membership = cv_membership_all[i_sub]
                print(len(cv_membership))
                fold_labels = []
                for i in range(len(cv_membership)):
                    fold_labels.append(np.mean(l_ints[cv_membership[i]]))

                tgm_rank = rank_fn(tgm_pred[i_sub, ...], fold_labels)
                multi_fold_acc.append(tgm_rank[None, ...])
            multi_fold_acc = np.concatenate(multi_fold_acc, axis=0)
            print(multi_fold_acc.shape)
            np.savez_compressed(rank_file, tgm_rank=multi_fold_acc)

        time = result['time']
        win_starts = result['win_starts']
        time_win = time[win_starts]

        mean_acc = np.mean(multi_fold_acc, axis=1)

        frac_above = np.sum(mean_acc > 0.5, axis=0)/float(mean_acc.shape[0])

        min_time = 0.0
        ax = combo_grid[i_word]
        im = ax.imshow(np.squeeze(frac_above), interpolation='nearest', aspect='auto',
                       vmin=0.5, vmax=1.0)

        ax.set_title('{word}'.format(
            word=PLOT_TITLE_WORD[word]), fontsize=axistitlesize)

        num_time = len(time_win)
        ax.set_xticks(np.arange(0, num_time, time_step) - time_adjust)

        max_time = 0.5 * len(time_win) / time_step
        label_time = np.arange(min_time, max_time, 0.5)

        ax.set_xticklabels(label_time)
        ax.set_yticks(np.arange(0, num_time, time_step) - time_adjust)
        ax.set_yticklabels(label_time)
        ax.tick_params(labelsize=ticklabelsize)
        ax.axvline(x=0.625*time_step, color='w')
        ax.text(-0.15, 1.05, string.ascii_uppercase[i_word], transform=ax.transAxes,
                size=axislettersize, weight='bold')

        cbar = combo_grid.cbar_axes[i_word].colorbar(im)
        time_adjust = args.win_len*0.002

    # for cax in combo_grid.cbar_axes:
    #     cax.toggle_label(False)

    combo_fig.suptitle('Fraction of Subject Folds above Chance\n{}'.format(PLOT_TITLE_SEN[sen_type]),
                       fontsize=suptitlesize)
    combo_fig.text(0.04, 0.275, 'Train Time Relative to Last Word Onset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    combo_fig.text(0.5, 0.04, 'Test Time Relative to Last Word Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm-frac_multisub{exc}_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=args.experiment, sen_type=sen_type, word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=args.win_len,
                overlap=args.overlap,
                num_instances=args.num_instances,
                exc=exc_str
            ), bbox_inches='tight')
    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm-frac_multisub{exc}_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=sen_type, word='all', alg=args.alg, avgTime=args.avgTime,
            avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            exc=exc_str,
            num_instances=args.num_instances
        ), bbox_inches='tight')
    plt.show()