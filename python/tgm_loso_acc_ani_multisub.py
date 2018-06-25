import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import string
from rank_from_pred import rank_from_pred
import os

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'

SAVE_FILE = '{dir}TGM-LOSO-ANI_multisub_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
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

    top_dir = TOP_DIR.format(exp=args.experiment)
    fname_orig = SAVE_FILE.format(dir=top_dir,
                                    win_len=args.win_len,
                                    ov=args.overlap,
                                    perm='F',
                                    alg=args.alg,
                                    adj=args.adj,
                                    avgTm=args.avgTime,
                                    avgTst=args.avgTest,
                                    inst=args.num_instances,
                                    rsP=1,
                                    mode='acc')
    result = np.load(fname_orig + '.npz')
    n1_time_all = result['n1_time']
    n2_time_all = result['n2_time']
    n1_win_starts_all = result['n1_win_starts']
    n2_win_starts_all = result['n2_win_starts']
    time = [[n1_time_all, n1_time_all], [n2_time_all, n1_time_all],
            [n1_time_all, n2_time_all], [n2_time_all, n2_time_all]]
    win_starts = [[n1_win_starts_all, n1_win_starts_all], [n2_win_starts_all, n1_win_starts_all],
                  [n1_win_starts_all, n2_win_starts_all], [n2_win_starts_all, n2_win_starts_all]]
    rank_file = SAVE_FILE.format(dir=top_dir,
                                  win_len=args.win_len,
                                  ov=args.overlap,
                                  perm='F',
                                  alg=args.alg,
                                  adj=args.adj,
                                  avgTm=args.avgTime,
                                  avgTst=args.avgTest,
                                  inst=args.num_instances,
                                  rsP=1,
                                  mode='rankacc')
    if os.path.isfile(rank_file + '.npz'):
        rank_result = np.load(rank_file + '.npz')
        acc_all = [result['tgm_rank_n1n1'],
                   result['tgm_rank_n1n2'],
                   result['tgm_rank_n2n1'],
                   result['tgm_rank_n2n2']]
    else:
        tgm_pred_n1n1 = result['tgm_pred_n1n1']
        tgm_pred_n1n2 = result['tgm_pred_n1n2']
        tgm_pred_n2n1 = result['tgm_pred_n2n1']
        tgm_pred_n2n2 = result['tgm_pred_n2n2']
        l_ints = result['l_ints']
        cv_membership = result['cv_membership']
        fold_labels = []
        for i in range(len(cv_membership)):
            fold_labels.append(np.mean(l_ints[cv_membership[i]]))

        tgm_rank_n1n1 = rank_from_pred(tgm_pred_n1n1, fold_labels)
        tgm_rank_n1n2 = rank_from_pred(tgm_pred_n1n2, fold_labels)
        tgm_rank_n2n1 = rank_from_pred(tgm_pred_n2n1, fold_labels)
        tgm_rank_n2n2 = rank_from_pred(tgm_pred_n2n2, fold_labels)
        np.savez_compressed(rank_file,
                            tgm_rank_n1n1=tgm_rank_n1n1,
                            tgm_rank_n1n2=tgm_rank_n1n2,
                            tgm_rank_n2n1=tgm_rank_n2n1,
                            tgm_rank_n2n2=tgm_rank_n2n2)
        acc_all = [tgm_rank_n1n1,
                   tgm_rank_n1n2,
                   tgm_rank_n2n1,
                   tgm_rank_n2n2]

    titles = ['Noun1-Noun1', 'Noun1-Noun2', 'Noun2-Noun1', 'Noun2-Noun2']

    time_step = int(50 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(12, 16))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, 2),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=False)
    for i_combo in range(len(time)):
        time_win_x = time[i_combo][0][win_starts[i_combo][0]]
        time_win_y = time[i_combo][1][win_starts[i_combo][1]]

        ax = combo_grid[i_combo]
        im = ax.imshow(np.mean(acc_all[i_combo], axis=0), interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.0)

        print(acc_all[i_combo].shape)
        print(len(time_win_x))

        ax.set_title(titles[i_combo], fontsize=axistitlesize)

        ax.set_xticks(np.arange(0.0, float(len(time_win_x)), float(time_step)) - time_adjust)
        ax.set_yticks(np.arange(0.0, float(len(time_win_y)), float(time_step)) - time_adjust)

        ax.set_xlim([0.0, float(len(time_win_x) - 1)])
        ax.set_ylim([float(len(time_win_y)) - 1, 0.0])

        min_time = 0.0
        max_time_x = 0.1 * len(time_win_x) / time_step
        label_time_x = np.arange(min_time, max_time_x, 0.1)
        ax.set_xticklabels(label_time_x)


        max_time_y = 0.1 * len(time_win_y) / time_step
        label_time_y = np.arange(min_time, max_time_y, 0.1)
        ax.set_yticklabels(label_time_y)

        ax.tick_params(labelsize=ticklabelsize)
        ax.text(-0.12, 1.02, string.ascii_uppercase[i_combo], transform=ax.transAxes,
                                size=axislettersize, weight='bold')

    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('Animacy Rank Accuracy TGMs',
        fontsize=suptitlesize)
    combo_fig.text(0.04, 0.275, 'Train Time Relative to Word Onset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    combo_fig.text(0.5, 0.04, 'Test Time Relative to Word Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.subplots_adjust(top=0.85)
    combo_fig.savefig('/home/nrafidi/thesis_figs/{exp}_avg-tgm-ani_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                    exp=args.experiment, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    num_instances=args.num_instances
                ), bbox_inches='tight')

    plt.show()