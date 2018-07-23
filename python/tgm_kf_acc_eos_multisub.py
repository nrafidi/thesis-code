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
from rank_from_pred import rank_from_pred

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


TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_KF_EOS/'
MULTI_SAVE_FILE = '{dir}TGM-LOSO-{k}F_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_rsCV{rsCV}_{rank_str}{mode}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', type=int, choices=[2,4,8])
    parser.add_argument('--cv_random_state', type=int, choices=range(100))
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='T')
    parser.add_argument('--avgTest', default='T')
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

    sen_type = 'pooled'
    experiment='krns2'
    num_folds = args.num_folds
    rsCV = args.cv_random_state
    word_list = ['verb', 'voice']
    if args.num_folds == 2:
        word_list.append('propid')

    n_rows=1
    num_plots = int(ceil(float(len(word_list))/float(n_rows)))
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(num_plots*6, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(n_rows, num_plots),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.2, share_all=True, label_mode='all')

    for i_word, word in enumerate(word_list):
        top_dir = TOP_DIR.format(exp=experiment)
        multi_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                            sen_type=sen_type,
                                            word=word,
                                            win_len=args.win_len,
                                            k=num_folds,
                                            rsCV=rsCV,
                                            ov=args.overlap,
                                            perm='F',
                                            alg=args.alg,
                                            adj=args.adj,
                                            avgTm=args.avgTime,
                                            avgTst=args.avgTest,
                                            inst=args.num_instances,
                                            rsP=1,
                                            rank_str='',
                                            mode='acc')
        rank_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                           sen_type=sen_type,
                                           word=word,
                                           win_len=args.win_len,
                                           k=num_folds,
                                           rsCV=rsCV,
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

        result = np.load(multi_file + '.npz')
        if os.path.isfile(rank_file + '.npz'):
            rank_result = np.load(rank_file + '.npz')
            multi_fold_acc = rank_result['tgm_rank']
        else:
            tgm_pred = result['tgm_pred']
            l_ints = result['l_ints']
            cv_membership = result['cv_membership']
            fold_labels = []
            for i in range(len(cv_membership)):
                fold_labels.append(l_ints[cv_membership[i]])

            multi_fold_acc = rank_from_pred(tgm_pred, fold_labels)
            np.savez_compressed(rank_file, tgm_rank=multi_fold_acc)

        time = result['time']
        win_starts = result['win_starts']
        time_win = time[win_starts]
        mean_acc = np.mean(multi_fold_acc, axis=0)

        min_time = 0.0

        ax = combo_grid[i_word]
        im = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto',
                       vmin=0.0, vmax=1.0)

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

    combo_fig.suptitle('Rank Accuracy TGMs {}-Fold\n{}'.format(num_folds, PLOT_TITLE_SEN[sen_type]),
                       fontsize=suptitlesize)
    combo_fig.text(0.04, 0.275, 'Train Time Relative to Last Word Onset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    combo_fig.text(0.5, 0.04, 'Test Time Relative to Last Word Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm-{k}F_{rsCV}_multisub_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=experiment, sen_type=sen_type, k=num_folds, rsCV=rsCV, word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=args.win_len,
                overlap=args.overlap,
                num_instances=args.num_instances,
            ), bbox_inches='tight')
    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm-{k}F_{rsCV}_multisub_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=experiment, sen_type=sen_type, k=num_folds, rsCV=rsCV, word='all', alg=args.alg, avgTime=args.avgTime,
            avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')
    plt.show()