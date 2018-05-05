import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO_det
from mpl_toolkits.axes_grid1 import AxesGrid
import string
from sklearn.metrics import f1_score


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_ANALYSIS = {'det-type-first': 'First Determiner ID',
                       'the-dog': 'The vs Dog'}

def intersect_accs(sen_type,
                   analysis,
                   win_len=100,
                   overlap=12,
                   alg='lr-l2',
                   adj='zscore',
                   num_instances=1,
                   avgTime='F',
                   avgTest='T'):
    top_dir = run_TGM_LOSO_det.TOP_DIR

    acc_by_sub = []
    acc_intersect = []
    time_by_sub = []
    win_starts_by_sub = []
    f1_by_sub = []
    for sub in run_TGM_LOSO_det.VALID_SUBS:
        save_dir = run_TGM_LOSO_det.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result_fname = run_TGM_LOSO_det.SAVE_FILE.format(dir=save_dir,
                                                           sub=sub,
                                                           sen_type=sen_type,
                                                           analysis=analysis,
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
            time = np.squeeze(result['time'])
            win_starts = result['win_starts']
            preds = result['tgm_pred']
            l_ints = result['l_ints']
            cv_membership = result['cv_membership']
        except:
            print(result_fname)
            continue

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
        fold_acc = result['tgm_acc']
        acc = np.mean(fold_acc, axis=0)
        time_by_sub.append(time[None, ...])
        win_starts_by_sub.append(win_starts[None, ...])
        acc_thresh = acc > 0.5
        acc_by_sub.append(acc[None, ...])
        f1_by_sub.append(tgm_f1[None, ...])
        acc_intersect.append(acc_thresh[None, ...])
    acc_all = np.concatenate(acc_by_sub, axis=0)
    f1_all = np.concatenate(f1_by_sub, axis=0)
    intersection = np.sum(np.concatenate(acc_intersect, axis=0), axis=0)
    time = np.mean(np.concatenate(time_by_sub, axis=0), axis=0)
    win_starts = np.mean(np.concatenate(win_starts_by_sub, axis=0), axis=0).astype('int')

    return intersection, acc_all, time, win_starts, f1_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sen_type', default='pooled', choices=run_TGM_LOSO_det.VALID_SEN_TYPE)
    parser.add_argument('--win_len', type=int, default=12)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=1)
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
    combo_fig = plt.figure(figsize=(15, 10))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(1, 2),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True)
    diag_fig, diag_ax = plt.subplots()
    for i_combo, analysis in enumerate(['det-type-first', 'the-dog']):
        intersection, acc_all, time, win_starts, f1 = intersect_accs(args.sen_type,
                                                                     analysis,
                                                                     win_len=args.win_len,
                                                                     overlap=args.overlap,
                                                                     alg=args.alg,
                                                                     adj=args.adj,
                                                                     num_instances=args.num_instances,
                                                                     avgTime=args.avgTime,
                                                                     avgTest=args.avgTest)
        frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
        mean_acc = np.mean(acc_all, axis=0)
        mean_f1 = np.mean(f1, axis=0)
        time_win = time[win_starts]
        # print(mean_acc.shape)
        print(np.max(np.diag(mean_f1)))
        num_time = len(time_win)

        ax = combo_grid[i_combo]
        im = ax.imshow(mean_f1, interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.0)
        if i_combo == 0:
            ax.set_ylabel('Train Time (s)')
        ax.set_xlabel('Test Time (s)')
        ax.set_title('{analysis} from {sen_type}'.format(
            sen_type=PLOT_TITLE_SEN[args.sen_type],
            analysis=PLOT_TITLE_ANALYSIS[analysis]), fontsize=14)

        ax.set_xticks(np.arange(0, num_time, time_step) - time_adjust)
        min_time = 0.0
        max_time = 0.5 * len(time_win) / time_step
        label_time = np.arange(min_time, max_time, 0.1)
        ax.set_xticklabels(label_time)
        ax.set_yticks(np.arange(0, num_time, time_step) - time_adjust)
        ax.set_yticklabels(label_time)

        ax.text(-0.15, 1.05, string.ascii_uppercase[i_combo], transform=ax.transAxes,
                                size=20, weight='bold')
        i_combo += 1

        diag_ax.plot(np.diag(mean_f1), label=PLOT_TITLE_ANALYSIS[analysis])

    diag_ax.set_xticks(np.arange(0, num_time, time_step) - time_adjust)
    min_time = 0.0
    max_time = 0.5 * num_time / time_step
    label_time = np.arange(min_time, max_time, 0.1)
    diag_ax.set_xticklabels(label_time)
    diag_ax.set_xlabel('Time from word onset (s)')
    diag_ax.legend()
    diag_fig.suptitle('F1 Scores Averaged Over Subjects',
                       fontsize=18)
    diag_fig.savefig(
        '/home/nrafidi/thesis_figs/krns2_diag-overlay-det_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            sen_type=args.sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('TGM of F1 Scores Averaged Over Subjects',
        fontsize=18)

    combo_fig.savefig('/home/nrafidi/thesis_figs/krns2_avg-tgm-det_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                    sen_type=args.sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    num_instances=args.num_instances
                ), bbox_inches='tight')

    plt.show()