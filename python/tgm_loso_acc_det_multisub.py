import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO_det_multisub_fold
from mpl_toolkits.axes_grid1 import AxesGrid
import string
from sklearn.metrics import f1_score


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_ANALYSIS = {'det-type-first': 'The vs A',
                       'the-dog': 'The vs Dog',
                       'a-dog': 'A vs Dog'}


def f1_calc(preds, l_ints, cv_membership, avgTest):
    num_fold, num_time, _ = preds.shape
    tgm_f1 = np.empty((num_time, num_time))
    for i_time in range(num_time):
        for j_time in range(num_time):
            pred_mat = []
            label_mat = []
            for i_fold in range(num_fold):
                p = preds[i_fold, i_time, j_time]
                pred_mat.append(np.argmax(p, axis=1))
                curr_labels = l_ints[cv_membership[i_fold, :]]
                if avgTest == 'T':
                    indexes = np.unique(curr_labels, return_index=True)[1]
                    meow = [curr_labels[index] for index in sorted(indexes)]
                    curr_labels = meow
                label_mat.append(curr_labels)
            pred_mat = np.concatenate(pred_mat, axis=0)
            label_mat = np.concatenate(label_mat, axis=0)
            tgm_f1[i_time, j_time] = f1_score(label_mat, pred_mat)
    return tgm_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sen_type', default='pooled', choices=run_TGM_LOSO_det_multisub_fold.VALID_SEN_TYPE)
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

    if args.avgTime == 'T':
        avg_time_str = 'Time Average'
    else:
        avg_time_str = 'No Time Average'

    if args.avgTest == 'T':
        avg_test_str = 'Test Sample Average'
    else:
        avg_test_str = 'No Test Sample Average'

    time_step = int(50 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(12, 8))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(1, 3),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True)
    diag_fig, diag_ax = plt.subplots()
    for i_combo, analysis in enumerate(['det-type-first', 'the-dog', 'a-dog']):

        multi_file = run_TGM_LOSO_det_multisub_fold.SAVE_FILE.format(dir=run_TGM_LOSO_det_multisub_fold.TOP_DIR,
                             sen_type=args.sen_type,
                             analysis=analysis,
                             win_len=args.win_len,
                             ov=args.overlap,
                             perm='F',
                             alg=args.alg,
                             adj=args.adj,
                             avgTm=args.avgTime,
                             avgTst=args.avgTest,
                             inst=args.num_instances,
                             rsP=1,
                             fold='acc')

        result = np.load(multi_file + '.npz')
        tgm_pred = result['tgm_pred']
        l_ints = result['l_ints']
        cv_membership = result['cv_membership']
        time = result['time']
        win_starts = result['win_starts']

        mean_f1 = f1_calc(preds=tgm_pred, l_ints=l_ints, cv_membership=cv_membership, avgTest=args.avgTest)

        time_win = time[win_starts]
        # print(mean_acc.shape)
        print(np.max(np.diag(mean_f1)))
        num_time = len(time_win)

        ax = combo_grid[i_combo]
        im = ax.imshow(mean_f1, interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.0)
        if i_combo == 0:
            ax.set_ylabel('Train Time (s)', fontsize=axislabelsize)
        # ax.set_xlabel('Test Time (s)')
        ax.set_title('{analysis}'.format(
            analysis=PLOT_TITLE_ANALYSIS[analysis]), fontsize=axistitlesize)

        ax.set_xticks(np.arange(0, num_time, time_step) - time_adjust)
        min_time = 0.0
        max_time = 0.5 * len(time_win) / time_step
        label_time = np.arange(min_time, max_time, 0.1)
        ax.set_xticklabels(label_time)
        ax.set_yticks(np.arange(0, num_time, time_step) - time_adjust)
        ax.set_yticklabels(label_time)
        ax.tick_params(labelsize=ticklabelsize)

        ax.text(-0.15, 1.05, string.ascii_uppercase[i_combo], transform=ax.transAxes,
                                size=axislettersize, weight='bold')
        i_combo += 1

        diag_ax.plot(np.diag(mean_f1), label=PLOT_TITLE_ANALYSIS[analysis])

    diag_ax.set_xticks(np.arange(0, num_time, time_step) - time_adjust)
    min_time = 0.0
    max_time = 0.5 * num_time / time_step
    label_time = np.arange(min_time, max_time, 0.1)
    diag_ax.set_xticklabels(label_time)
    diag_ax.set_xlabel('Time from Word Onset (s)', fontsize=16)
    diag_ax.legend(loc=4, fontsize=legendfontsize)
    diag_ax.tick_params(labelsize=ticklabelsize)
    diag_fig.suptitle('F1 Scores Averaged Over Subjects',
                       fontsize=suptitlesize)
    diag_fig.savefig(
        '/home/nrafidi/thesis_figs/krns2_diag-overlay-det_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            sen_type=args.sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.text(0.5, 0.04, 'Test Time (s)', ha='center', fontsize=axislabelsize)
    combo_fig.suptitle('TGM of F1 Scores Averaged Over Subjects',
        fontsize=suptitlesize)

    combo_fig.savefig('/home/nrafidi/thesis_figs/krns2_avg-tgm-det_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                    sen_type=args.sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    num_instances=args.num_instances
                ), bbox_inches='tight')

    plt.show()