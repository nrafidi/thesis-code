import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO_animacy_fold as run_TGM_LOSO
from mpl_toolkits.axes_grid1 import AxesGrid
import string

SAVE_FILE = '{dir}TGM-LOSO-ANI_{sub}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'


def intersect_accs(exp,
                   win_len=100,
                   overlap=12,
                   alg='lr-l2',
                   adj=None,
                   num_instances=1,
                   avgTime='F',
                   avgTest='F'):
    top_dir = run_TGM_LOSO.TOP_DIR.format(exp=exp)

    n1n1_acc_by_sub = []
    n1n2_acc_by_sub = []
    n2n1_acc_by_sub = []
    n2n2_acc_by_sub = []
    n1_time_by_sub = []
    n2_time_by_sub = []
    n1_win_starts_by_sub = []
    n2_win_starts_by_sub = []
    for sub in run_TGM_LOSO.VALID_SUBS[exp]:
        save_dir = run_TGM_LOSO.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result_fname = SAVE_FILE.format(dir=save_dir,
                                       sub=sub,
                                       win_len=win_len,
                                       ov=overlap,
                                       perm='F',
                                       alg=alg,
                                       adj=adj,
                                       avgTm=avgTime,
                                       avgTst=avgTest,
                                       inst=num_instances,
                                       rsP=1,
                                       mode='acc') + '.npz'
        if not os.path.isfile(result_fname):
            print(result_fname)
            continue
        try:
            result = np.load(result_fname)
            n1_time = np.squeeze(result['n1_time'])
            n2_time = np.squeeze(result['n2_time'])
            n1_win_starts = result['n1_win_starts']
            n2_win_starts = result['n2_win_starts']
        except:
            print(result_fname)
            continue

        n1n1_acc = np.mean(result['tgm_acc_n1n1'], axis=0)
        n1n2_acc = np.mean(result['tgm_acc_n1n2'], axis=0)
        n2n1_acc = np.mean(result['tgm_acc_n2n1'], axis=0)
        n2n2_acc = np.mean(result['tgm_acc_n2n2'], axis=0)

        n1_time_by_sub.append(n1_time[None, ...])
        n2_time_by_sub.append(n2_time[None, ...])

        n1_win_starts_by_sub.append(n1_win_starts[None, ...])
        n2_win_starts_by_sub.append(n2_win_starts[None, ...])

        n1n1_acc_by_sub.append(n1n1_acc[None, ...])
        n1n2_acc_by_sub.append(n1n2_acc[None, ...])
        n2n1_acc_by_sub.append(n2n1_acc[None, ...])
        n2n2_acc_by_sub.append(n2n2_acc[None, ...])

    n1n1_acc_all = np.concatenate(n1n1_acc_by_sub, axis=0)
    n1n2_acc_all = np.concatenate(n1n2_acc_by_sub, axis=0)
    n2n1_acc_all = np.concatenate(n2n1_acc_by_sub, axis=0)
    n2n2_acc_all = np.concatenate(n2n2_acc_by_sub, axis=0)

    acc_all = [n1n1_acc_all,
              n1n2_acc_all,
              n2n1_acc_all,
              n2n2_acc_all]

    n1_time_all = np.mean(np.concatenate(n1_time_by_sub, axis=0), axis=0)
    n2_time_all = np.mean(np.concatenate(n2_time_by_sub, axis=0), axis=0)
    time = [[n1_time_all, n1_time_all], [n2_time_all, n1_time_all],
            [n1_time_all, n2_time_all], [n2_time_all, n2_time_all]]

    n1_win_starts_all = np.mean(np.concatenate(n1_win_starts_by_sub, axis=0), axis=0).astype('int')
    n2_win_starts_all = np.mean(np.concatenate(n2_win_starts_by_sub, axis=0), axis=0).astype('int')

    win_starts = [[n1_win_starts_all, n1_win_starts_all], [n1_win_starts_all, n2_win_starts_all],
            [n2_win_starts_all, n1_win_starts_all], [n2_win_starts_all, n2_win_starts_all]]

    return acc_all, time, win_starts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=10)
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='T')
    args = parser.parse_args()

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    acc_all, time, win_starts = intersect_accs(args.experiment,
                                              win_len=args.win_len,
                                              overlap=args.overlap,
                                              alg=args.alg,
                                              adj=args.adj,
                                              num_instances=args.num_instances,
                                              avgTime=args.avgTime,
                                              avgTest=args.avgTest)

    titles = ['Noun1-Noun1', 'Noun1-Noun2', 'Noun2-Noun1', 'Noun2-Noun2']

    time_step = int(50 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(16, 16))
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
        ax.set_yticks(np.arange(0.0, float(len(time_win_y)), time_step) - time_adjust)

        ax.set_xlim([0.0, float(len(time_win_x))])
        ax.set_ylim([float(len(time_win_y)), 0.0])

        min_time = 0.0
        max_time_x = 0.1 * len(time_win_x) / time_step
        label_time_x = np.arange(min_time, max_time_x, 0.5)
        ax.set_xticklabels(label_time_x)


        max_time_y = 0.1 * len(time_win_y) / time_step
        label_time_y = np.arange(min_time, max_time_y, 0.5)
        ax.set_yticklabels(label_time_y)

        ax.tick_params(labelsize=ticklabelsize)
        ax.text(-0.12, 1.02, string.ascii_uppercase[i_combo], transform=ax.transAxes,
                                size=axislettersize, weight='bold')

    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('Animacy TGM Averaged Over Subjects',
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