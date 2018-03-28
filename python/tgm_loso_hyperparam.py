import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import scipy.io as sio
from scipy.stats import zscore
import run_TGM_LOSO
import tgm_loso_acc

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_TGM_LOSO.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['noun1', 'verb', 'noun2'])
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='F')
    parser.add_argument('--percentile', type=float, default=0.1)
    args = parser.parse_args()

    if args.sen_type == 'active':
        max_time = 2.0
    else:
        max_time = 3.0

    win_lens = [12, 25, 50, 100, 150]
    num_insts = [1, 2, 5, 10]

    perc = args.percentile

    frac_sub_eos = []
    frac_sub_tot = []
    mean_max_eos = []
    mean_max_tot = []
    mean_perc_eos = []
    mean_perc_tot = []
    per_sub_max_eos = []
    per_sub_max_tot = []
    per_sub_perc_tot = []
    per_sub_perc_eos = []
    for win_len in win_lens:
        time_adjust = win_len * 0.002

        frac_sub_eos_win = []
        frac_sub_tot_win = []
        mean_max_eos_win = []
        mean_max_tot_win = []
        mean_perc_eos_win = []
        mean_perc_tot_win = []
        per_sub_max_eos_win = []
        per_sub_max_tot_win = []
        per_sub_perc_tot_win = []
        per_sub_perc_eos_win = []
        for num_instances in num_insts:
            intersection, acc_all, time, win_starts, eos_max = tgm_loso_acc.intersect_accs(args.experiment,
                                                                                           args.sen_type,
                                                                                           args.word,
                                                                                           win_len=win_len,
                                                                                           overlap=12,
                                                                                           adj=args.adj,
                                                                                           num_instances=num_instances,
                                                                                           avgTime=args.avgTime,
                                                                                           avgTest=args.avgTest)
            time = np.squeeze(time)
            time_ind = np.where(time[win_starts] >= (max_time - time_adjust))
            time_ind = time_ind[0]

            frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
            mean_acc = np.diag(np.mean(acc_all, axis=0))

            percentile_tot = int(perc * len(mean_acc))
            # print(percentile_tot)
            percentile_eos = int(perc * len(time_ind))
            # print(percentile_eos)

            mean_max_tot_win.append(np.max(mean_acc))
            sorted_tot_acc = np.sort(mean_acc)[::-1]
            mean_perc_tot_win.append(sorted_tot_acc[percentile_tot])
            frac_sub_tot_win.append(frac_sub[np.argmax(mean_acc)])

            mean_max_eos_win.append(np.max(mean_acc[time_ind]))
            sorted_eos_acc = np.sort(mean_acc[time_ind])[::-1]
            mean_perc_eos_win.append(sorted_eos_acc[percentile_eos])
            frac_sub_eos_win.append(frac_sub[time_ind[np.argmax(mean_acc[time_ind])]])

            sub_eos_max = []
            sub_eos_perc = []
            sub_tot_max = []
            sub_tot_perc = []
            for i_sub in range(acc_all.shape[0]):
                diag_acc = np.diag(np.squeeze(acc_all[i_sub, :, :]))
                sub_tot_max.append(np.max(diag_acc))
                sorted_diag_acc = np.sort(diag_acc)[::-1]
                sub_tot_perc.append(sorted_diag_acc[percentile_tot])
                sub_eos_max.append(np.max(diag_acc[time_ind]))
                sorted_eos_diag_acc = np.sort(diag_acc[time_ind])[::-1]
                sub_eos_perc.append(sorted_eos_diag_acc[percentile_eos])
            sub_tot_max = np.array(sub_tot_max)
            sub_tot_perc = np.array(sub_tot_perc)
            sub_eos_max = np.array(sub_eos_max)
            sub_eos_perc = np.array(sub_eos_perc)

            per_sub_max_eos_win.append(sub_eos_max[None, ...])
            per_sub_max_tot_win.append(sub_tot_max[None, ...])
            per_sub_perc_tot_win.append(sub_tot_perc[None, ...])
            per_sub_perc_eos_win.append(sub_eos_perc[None, ...])

        frac_sub_eos_win = np.array(frac_sub_eos_win)
        frac_sub_eos.append(frac_sub_eos_win[None, ...])

        frac_sub_tot_win = np.array(frac_sub_tot_win)
        frac_sub_tot.append(frac_sub_tot_win[None, ...])

        mean_max_eos_win = np.array(mean_max_eos_win)
        mean_max_eos.append(mean_max_eos_win[None, ...])

        mean_max_tot_win = np.array(mean_max_tot_win)
        mean_max_tot.append(mean_max_tot_win[None, ...])

        mean_perc_eos_win = np.array(mean_perc_eos_win)
        mean_perc_eos.append(mean_perc_eos_win[None, ...])

        mean_perc_tot_win = np.array(mean_perc_tot_win)
        mean_perc_tot.append(mean_perc_tot_win[None, ...])

        per_sub_max_eos_win = np.concatenate(per_sub_max_eos_win, axis=0)
        per_sub_max_eos.append(per_sub_max_eos_win[None, ...])

        per_sub_max_tot_win = np.concatenate(per_sub_max_tot_win, axis=0)
        per_sub_max_tot.append(per_sub_max_tot_win[None, ...])

        per_sub_perc_tot_win = np.concatenate(per_sub_perc_tot_win, axis=0)
        per_sub_perc_tot.append(per_sub_perc_tot_win[None, ...])

        per_sub_perc_eos_win = np.concatenate(per_sub_perc_eos_win, axis=0)
        per_sub_perc_eos.append(per_sub_perc_eos_win[None, ...])

    frac_sub_eos = np.concatenate(frac_sub_eos, axis=0)

    frac_sub_tot = np.concatenate(frac_sub_tot, axis=0)

    mean_max_eos = np.concatenate(mean_max_eos, axis=0)

    mean_max_tot = np.concatenate(mean_max_tot, axis=0)

    mean_perc_eos = np.concatenate(mean_perc_eos, axis=0)

    mean_perc_tot = np.concatenate(mean_perc_tot, axis=0)

    per_sub_max_eos = np.concatenate(per_sub_max_eos, axis=0)

    per_sub_max_tot = np.concatenate(per_sub_max_tot, axis=0)

    per_sub_perc_eos = np.concatenate(per_sub_perc_eos, axis=0)

    per_sub_perc_tot = np.concatenate(per_sub_perc_tot, axis=0)

    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    h00 = axs[0][0].imshow(frac_sub_tot, interpolation='nearest', vmin=0.5, vmax=1.0)
    axs[0][0].set_title('Global Fraction of Subjects > Chance')
    fig.colorbar(h00, ax=axs[0][0], shrink=0.8)
    h01 = axs[0][1].imshow(frac_sub_eos, interpolation='nearest', vmin=0.5, vmax=1.0)
    axs[0][1].set_title('Post-Sentence Fraction of Subjects > Chance')
    fig.colorbar(h01, ax=axs[0][1], shrink=0.8)
    h10 = axs[1][0].imshow(mean_max_tot, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[1][0].set_title('Global Max Accuracy')
    fig.colorbar(h10, ax=axs[1][0], shrink=0.8)
    h11 = axs[1][1].imshow(mean_max_eos, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[1][1].set_title('Post-Sentence Max Accuracy')
    fig.colorbar(h11, ax=axs[1][1], shrink=0.8)
    h20 = axs[2][0].imshow(mean_perc_tot, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[2][0].set_title('Global {}th Percentile'.format(int(perc*100)))
    fig.colorbar(h20, ax=axs[2][0], shrink=0.8)
    h21 = axs[2][1].imshow(mean_perc_eos, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[2][1].set_title('Post-Sentence {}th Percentile'.format(int(perc*100)))
    fig.colorbar(h21, ax=axs[2][1], shrink=0.8)

    for i in range(3):
        for j in range(2):
            axs[i][j].set_xticks(range(len(num_insts)))
            axs[i][j].set_xticklabels(num_insts)
            axs[i][j].set_yticks(range(len(win_lens)))
            axs[i][j].set_yticklabels(np.array(win_lens).astype('float') * 2)
            if i == 2:
                axs[i][j].set_xlabel('Number of Instances')
            if j == 0:
                axs[i][j].set_ylabel('Window Length (ms)')


    fig.suptitle('Accuracy and Consistency Scores\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                             args.word,
                                                                             args.avgTime,
                                                                             args.avgTest),
                 fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(top=0.87)

    fig = plt.figure()
    grid = AxesGrid(fig, 111, nrows_ncols=(1, 2),
                    axes_pad=0.3, cbar_mode='single', cbar_location='right',
                    cbar_pad=0.1)

    mats_to_plot = [zscore(frac_sub_tot) + zscore(mean_perc_tot), zscore(frac_sub_eos) + zscore(mean_perc_eos)]
    titles = ['Combined Global Score', 'Combined Post-Sentence Score']
    for i_ax, ax in enumerate(grid):
        im = ax.imshow(mats_to_plot[i_ax], interpolation='nearest', vmin=0.0,
                       vmax=3.0)
        ax.set_title(titles[i_ax])
        ax.set_xticks(range(len(num_insts)))
        ax.set_xticklabels(num_insts)
        ax.set_yticks(range(len(win_lens)))
        ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
        ax.set_xlabel('Number of Instances')
        if i_ax == 0:
            ax.set_ylabel('Window Length (ms)')

    cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle('Combined Score\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                      args.word,
                                                                      args.avgTime,
                                                                      args.avgTest),
                 fontsize=24)

    num_sub = per_sub_max_tot.shape[2]
    half_sub = int(num_sub/2)
    fig = plt.figure(figsize=(30, 10))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, half_sub),
                    axes_pad=0.3, cbar_mode='single', cbar_location='right',
                    cbar_pad=0.1)

    for i_ax, ax in enumerate(grid):
        im = ax.imshow(np.squeeze(per_sub_max_tot[:, :, i_ax]), interpolation='nearest', vmin=0.25,
                       vmax=1.0)
        ax.set_title(run_TGM_LOSO.VALID_SUBS[args.experiment][i_ax])
        ax.set_xticks(range(len(num_insts)))
        ax.set_xticklabels(num_insts)
        ax.set_yticks(range(len(win_lens)))
        ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
        if i_ax >= half_sub:
            ax.set_xlabel('Number of Instances')
        if i_ax == half_sub or i_ax == 0:
            ax.set_ylabel('Window Length (ms)')

    cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle('Per Subject Global Max\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                      args.word,
                                                                      args.avgTime,
                                                                      args.avgTest),
                 fontsize=24)

    num_sub = per_sub_max_eos.shape[2]
    half_sub = int(num_sub / 2)

    fig = plt.figure(figsize = (30, 10))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, half_sub),
                    axes_pad=0.3, cbar_mode='single', cbar_location='right',
                    cbar_pad=0.1)

    for i_ax, ax in enumerate(grid):
        im = ax.imshow(np.squeeze(per_sub_max_eos[:, :, i_ax]), interpolation='nearest', vmin=0.25,
                                   vmax=1.0)
        ax.set_title(run_TGM_LOSO.VALID_SUBS[args.experiment][i_ax])
        ax.set_xticks(range(len(num_insts)))
        ax.set_xticklabels(num_insts)
        ax.set_yticks(range(len(win_lens)))
        ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
        if i_ax >= half_sub:
            ax.set_xlabel('Number of Instances')
        if i_ax == half_sub or i_ax == 0:
            ax.set_ylabel('Window Length (ms)')

    cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle('Per Subject Post-Sentence Max\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                              args.word,
                                                                              args.avgTime,
                                                                              args.avgTest),
                 fontsize=24)

    plt.show()
    #
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_win_inst_comp_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest
    #     ), bbox_inches='tight')
    #
    #
    # fig, ax = plt.subplots()
    # h0 = ax.imshow(frac_sub_tot + mean_acc_tot, interpolation='nearest', vmin=0.75, vmax=2.0)
    # ax.set_xticks(range(len(num_insts)))
    # ax.set_xticklabels(num_insts)
    # ax.set_yticks(range(len(win_lens)))
    # ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
    # ax.set_title('Frac Subjects > Chance + Max EOS Accuracy')
    # ax.set_xlabel('Number of Instances')
    # ax.set_ylabel('Window Length (ms)')
    # fig.colorbar(h0, ax=ax)
    # fig.suptitle('Post Sentence Maximum\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
    #                                                                          args.word,
    #                                                                          args.avgTime,
    #                                                                          args.avgTest))
    # plt.subplots_adjust(top=0.8)
    #
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_win_inst_comp-sum_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest
    #     ), bbox_inches='tight')
    # print(arg_max_eos)
    # print(arg_max_tot)
    # print(np.squeeze(per_sub_max_eos[:, 0, :]))
    #
    #
    #
    # plt.show()


