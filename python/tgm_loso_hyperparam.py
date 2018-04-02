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
import string

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='F')
    parser.add_argument('--percentile', type=float, default=0.1)
    args = parser.parse_args()

    

    if args.avgTime == 'T':
        avg_time_str = 'Time Average'
    else:
        avg_time_str = 'No Time Average'

    if args.avgTest == 'T':
        avg_test_str = 'Test Sample Average'
    else:
        avg_test_str = 'No Test Sample Average'

    win_lens = [12, 25, 50, 100, 150]
    num_insts = [2, 5, 10]

    perc = args.percentile

    fig_fname = '/home/nrafidi/thesis_figs/{exp}_{fig_type}_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}_perc{perc}.pdf'
    combo_scores = []
    combo_fig = plt.figure()
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, 2),
                        axes_pad=0.3, cbar_mode='single', cbar_location='right',
                        cbar_pad=0.1)
    i_combo = 0
    for sen_type in ['active', 'passive']:
        for word in ['noun1', 'verb']:
            if sen_type == 'active':
                max_time = 2.0
            else:
                max_time = 3.0

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
                                                                                                   sen_type,
                                                                                                   word,
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

            fig = plt.figure()
            grid = AxesGrid(fig, 111, nrows_ncols=(2, 2),
                            axes_pad=0.3, cbar_mode='single', cbar_location='right',
                            cbar_pad=0.1)

            mats_to_plot = [frac_sub_tot, frac_sub_eos, mean_max_tot, mean_max_eos]
            titles = ['Global Fraction of Subjects > Chance', 'Post-Sentence Fraction of Subjects > Chance',
                      'Global Max Accuracy', 'Post-Sentence Max Accuracy']
            for i_ax, ax in enumerate(grid):
                im = ax.imshow(mats_to_plot[i_ax], interpolation='nearest', vmin=0.25,
                                   vmax=1.0)
                ax.set_title(titles[i_ax])
                ax.set_xticks(range(len(num_insts)))
                ax.set_xticklabels(num_insts)
                ax.set_yticks(range(len(win_lens)))
                ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
                ax.text(-0.3, 1.1, string.ascii_uppercase[i_ax], transform=ax.transAxes,
                                    size=20, weight='bold')
                if i_ax > 1:
                    ax.set_xlabel('Number of Instances')
                if i_ax == 0 or i_ax == 2:
                    ax.set_ylabel('Window Length (ms)')
            cbar = grid.cbar_axes[0].colorbar(im)
            fig.suptitle('Accuracy and Consistency Scores\nDecoding {word} from {sen}, {avgTime}, {avgTest}'.format(sen = PLOT_TITLE_SEN[sen_type],
                                                                                     word=PLOT_TITLE_WORD[word],
                                                                                     avgTime=avg_time_str,
                                                                                     avgTest=avg_test_str),
                         fontsize=18)

            fig.savefig(fig_fname.format(
                        exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
                        perc=perc, fig_type='single-mean-score-comp'
                    ), bbox_inches='tight')

            z_frac = zscore(frac_sub_tot)
            z_frac[np.isnan(z_frac)] = 0.0
            z_frac_eos = zscore(frac_sub_eos)
            z_frac_eos[np.isnan(z_frac_eos)] = 0.0
            all_combined = (z_frac_eos + zscore(mean_max_eos))/2.0
            all_combined_z = zscore(all_combined)
            combo_scores.append(all_combined_z[None, ...])

            fig, ax = plt.subplots()
            h = ax.imshow(all_combined, interpolation='nearest', aspect='auto', vmin=-3.0, vmax=3.0)
            plt.colorbar(h)
            ax.set_title('Combined Score\nDecoding {word} from {sen}\n{avgTime}, {avgTest}'.format(sen = PLOT_TITLE_SEN[sen_type],
                                                                                     word=PLOT_TITLE_WORD[word],
                                                                                     avgTime=avg_time_str,
                                                                                     avgTest=avg_test_str),
                         fontsize=14)
            ax.set_xticks(range(len(num_insts)))
            ax.set_xticklabels(num_insts)
            ax.set_yticks(range(len(win_lens)))
            ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
            ax.set_xlabel('Number of Instances')
            ax.set_ylabel('Window Length (ms)')
            fig.tight_layout()
            plt.savefig(fig_fname.format(
                exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
                perc=perc, fig_type='total-comb-max-score-comp'
            ), bbox_inches='tight')

            im = combo_grid[i_combo].imshow(all_combined, interpolation='nearest', vmin=-3.0, vmax=3.0)
            combo_grid[i_combo].set_title('Decoding {word} from {sen}'.format(sen = PLOT_TITLE_SEN[sen_type],
                                                                              word=PLOT_TITLE_WORD[word]))
            combo_grid[i_combo].set_xticks(range(len(num_insts)))
            combo_grid[i_combo].set_xticklabels(num_insts)
            combo_grid[i_combo].set_yticks(range(len(win_lens)))
            combo_grid[i_combo].set_yticklabels(np.array(win_lens).astype('float') * 2)
            combo_grid[i_combo].text(-0.3, 1.1, string.ascii_uppercase[i_ax], transform=ax.transAxes,
                                    size=20, weight='bold')
            if i_combo > 1:
                combo_grid[i_combo].set_xlabel('Number of Instances')
            if i_combo == 0 or i_combo == 2:
                combo_grid[i_combo].set_ylabel('Window Length (ms)')
            i_combo += 1

    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('Combined Scores\n{avgTime}, {avgTest}'.format(
        avgTime=avg_time_str,
        avgTest=avg_test_str),
                 fontsize=18)

    combo_fig.savefig(fig_fname.format(
        exp=args.experiment, sen_type='both', word='all', avgTime=args.avgTime, avgTest=args.avgTest,
        perc=perc, fig_type='word-sen-combo-max-score-comp'
    ), bbox_inches='tight')


    all_combined = np.sum(np.concatenate(combo_scores, axis=0), axis=0)
    optimal = np.unravel_index(np.argmax(all_combined), all_combined.shape)

    fig, ax = plt.subplots()
    h = ax.imshow(all_combined, interpolation='nearest', aspect='auto', vmin=-7.0, vmax=7.0)
    plt.colorbar(h)
    ax.set_title('Total Combined Score\n{avgTime}, {avgTest}'.format(
        avgTime=avg_time_str,
        avgTest=avg_test_str),
                 fontsize=14)
    ax.set_xticks(range(len(num_insts)))
    ax.set_xticklabels(num_insts)
    ax.set_yticks(range(len(win_lens)))
    ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
    ax.set_xlabel('Number of Instances')
    ax.set_ylabel('Window Length (ms)')
    fig.tight_layout()
    plt.savefig(fig_fname.format(
        exp=args.experiment, sen_type='both', word='all', avgTime=args.avgTime, avgTest=args.avgTest,
        perc=perc, fig_type='total-comb-max-score-comp'
    ), bbox_inches='tight')

    print(
    'Optimal window size: {win}\nOptimal number of instances: {ni}\nScore: {score}'.format(win=win_lens[optimal[0]],
                                                                                           ni=num_insts[optimal[1]],
                                                                                           score=np.max(all_combined)))

    plt.show()
    #


    # fig = plt.figure()
    # grid = AxesGrid(fig, 111, nrows_ncols=(1, 2),
    #                 axes_pad=0.3, cbar_mode='single', cbar_location='right',
    #                 cbar_pad=0.1)
    #
    # z_frac = zscore(frac_sub_tot)
    # z_frac[np.isnan(z_frac)] = 0.0
    #
    # mats_to_plot = [z_frac + zscore(mean_max_tot), zscore(frac_sub_eos) + zscore(mean_max_eos)]
    # # vmin = np.min([np.min(mat) for mat in mats_to_plot])
    # # vmax = np.max([np.max(mat) for mat in mats_to_plot])
    # vmin = -4.0
    # vmax = 4.0
    # titles = ['Global', 'Post-Sentence']
    # for i_ax, ax in enumerate(grid):
    #     im = ax.imshow(mats_to_plot[i_ax], interpolation='nearest', vmin=vmin,
    #                    vmax=vmax)
    #     ax.set_title(titles[i_ax])
    #     ax.set_xticks(range(len(num_insts)))
    #     ax.set_xticklabels(num_insts)
    #     ax.set_yticks(range(len(win_lens)))
    #     ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
    #     ax.set_xlabel('Number of Instances')
    #     if i_ax == 0:
    #         ax.set_ylabel('Window Length (ms)')
    #
    # cbar = grid.cbar_axes[0].colorbar(im)
    # fig.subplots_adjust(top=0.8)
    # fig.suptitle(
    #     'Combined Max Score\nDecoding {word} from {sen}\n{avgTime}, {avgTest}'.format(sen=PLOT_TITLE_SEN[sen_type],
    #                                                                                   word=PLOT_TITLE_WORD[word],
    #                                                                                   avgTime=avg_time_str,
    #                                                                                   avgTest=avg_test_str),
    #     fontsize=16)
    #
    # plt.savefig(fig_fname.format(
    #     exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
    #     perc=perc, fig_type='comb-max-score-comp'
    # ), bbox_inches='tight')


    # mats_to_plot = [z_frac + zscore(mean_perc_tot), zscore(frac_sub_eos) + zscore(mean_perc_eos)]
    # # vmin = np.min([np.min(mat) for mat in mats_to_plot])
    # # vmax = np.max([np.max(mat) for mat in mats_to_plot])
    # vmin = -4.0
    # vmax = 4.0
    # titles = ['Global', 'Post-Sentence']
    # for i_ax, ax in enumerate(grid):
    #     im = ax.imshow(mats_to_plot[i_ax], interpolation='nearest', vmin=vmin,
    #                    vmax=vmax)
    #     ax.set_title(titles[i_ax])
    #     ax.set_xticks(range(len(num_insts)))
    #     ax.set_xticklabels(num_insts)
    #     ax.set_yticks(range(len(win_lens)))
    #     ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
    #     ax.set_xlabel('Number of Instances')
    #     if i_ax == 0:
    #         ax.set_ylabel('Window Length (ms)')
    #
    # cbar = grid.cbar_axes[0].colorbar(im)
    # fig.subplots_adjust(top=0.8)
    # fig.suptitle('Combined Percentile Score\nDecoding {word} from {sen}\n{avgTime}, {avgTest}'.format(sen = PLOT_TITLE_SEN[sen_type],
    #                                                                          word=PLOT_TITLE_WORD[word],
    #                                                                          avgTime=avg_time_str,
    #                                                                          avgTest=avg_test_str),
    #              fontsize=16)
    #
    # plt.savefig(fig_fname.format(
    #     exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
    #     perc=perc, fig_type='comb-perc-score-comp'
    # ), bbox_inches='tight')

    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_win_inst_comp_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest
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
    # fig.suptitle('Post Sentence Maximum\n{} {} avgTime {} avgTest {}'.format(sen_type,
    #                                                                          word,
    #                                                                          args.avgTime,
    #                                                                          args.avgTest))
    # plt.subplots_adjust(top=0.8)
    #
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_win_inst_comp-sum_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest
    #     ), bbox_inches='tight')
    # print(arg_max_eos)
    # print(arg_max_tot)
    # print(np.squeeze(per_sub_max_eos[:, 0, :]))
    #
    #
    #
    # plt.show()

# num_sub = per_sub_max_eos.shape[2]
# half_sub = int(num_sub / 2)

# fig = plt.figure(figsize = (30, 10))
# grid = AxesGrid(fig, 111, nrows_ncols=(2, half_sub),
#                 axes_pad=0.3, cbar_mode='single', cbar_location='right',
#                 cbar_pad=0.1)
#
# for i_ax, ax in enumerate(grid):
#     im = ax.imshow(np.squeeze(per_sub_max_eos[:, :, i_ax]), interpolation='nearest', vmin=0.25,
#                                vmax=1.0)
#     ax.set_title(run_TGM_LOSO.VALID_SUBS[args.experiment][i_ax])
#     ax.set_xticks(range(len(num_insts)))
#     ax.set_xticklabels(num_insts)
#     ax.set_yticks(range(len(win_lens)))
#     ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
#     if i_ax >= half_sub:
#         ax.set_xlabel('Number of Instances')
#     if i_ax == half_sub or i_ax == 0:
#         ax.set_ylabel('Window Length (ms)')
#
# cbar = grid.cbar_axes[0].colorbar(im)
# fig.suptitle('Per Subject Post-Sentence Max\nDecoding {word} from {sen}, {avgTime}, {avgTest}'.format(sen = PLOT_TITLE_SEN[sen_type],
#                                                                          word=PLOT_TITLE_WORD[word],
#                                                                          avgTime=avg_time_str,
#                                                                          avgTest=avg_test_str),
#              fontsize=18)
#
# plt.savefig(fig_fname.format(
#     exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
#     perc=perc, fig_type='sub-post-max-comp'
# ), bbox_inches='tight')
