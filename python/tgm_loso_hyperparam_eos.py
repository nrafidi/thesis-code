import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import run_TGM_LOSO_EOS
import tgm_loso_acc_eos
from scipy.stats import zscore
from mpl_toolkits.axes_grid1 import AxesGrid
import string

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'
PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'propid': 'Proposition ID',
                   'voice': 'Sentence Voice'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='T')
    args = parser.parse_args()

    sen_type_list = ['pooled', 'active', 'passive']

    if args.avgTime == 'T':
        avg_time_str = 'Time Average'
    else:
        avg_time_str = 'No Time Average'

    if args.avgTest == 'T':
        avg_test_str = 'Test Sample Average'
    else:
        avg_test_str = 'No Test Sample Average'

    win_lens = [12, 25, 50, 100]
    num_insts = [2, 5, 10]

    fig_fname = '/home/nrafidi/thesis_figs/{exp}_eos_{fig_type}_{sen_type}_{word}_{alg}_avgTime{avgTime}_avgTest{avgTest}.pdf'
    combo_scores = []
    for j_sen, sen_type in enumerate(sen_type_list):
        print(sen_type)
        sen_combo_scores = []
        if sen_type == 'pooled':
            word_list = ['noun1', 'verb', 'agent', 'patient', 'voice', 'propid']
            num_plots = 3
        else:
            word_list = ['verb', 'agent', 'patient']
            num_plots = 2

        combo_fig = plt.figure(figsize=(12, 12))
        combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, num_plots),
                              axes_pad=0.7, cbar_mode='single', cbar_location='right',
                              cbar_pad=0.5)
        for i_word, word in enumerate(word_list):
            print(word)
            chance = tgm_loso_acc_eos.CHANCE[args.experiment][sen_type][word]
            frac_sub_eos = []
            mean_max_eos = []
            for win_len in win_lens:
                time_adjust = win_len * 0.002
                frac_sub_win = []
                mean_max_win = []
                for num_instances in num_insts:
                    intersection, acc_all, time, win_starts, eos_max = tgm_loso_acc_eos.intersect_accs(args.experiment,
                                                                                                   sen_type,
                                                                                                   word,
                                                                                                   win_len=win_len,
                                                                                                   overlap=12,
                                                                                                   alg=args.alg,
                                                                                                   adj=args.adj,
                                                                                                   num_instances=num_instances,
                                                                                                   avgTime=args.avgTime,
                                                                                                   avgTest=args.avgTest)

                    frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
                    mean_acc = np.diag(np.mean(acc_all, axis=0))

                    mean_max_win.append(np.max(mean_acc))
                    argmax_mean_win = np.argmax(mean_acc)
                    frac_sub_win.append(frac_sub[argmax_mean_win])

                frac_sub_win = np.array(frac_sub_win)
                mean_max_win = np.array(mean_max_win)

                frac_sub_eos.append(frac_sub_win[None, ...])
                mean_max_eos.append(mean_max_win[None, ...])

            frac_sub_eos = np.concatenate(frac_sub_eos, axis=0)
            mean_max_eos = np.concatenate(mean_max_eos, axis=0)

            # fig = plt.figure(figsize=(12, 12))
            # grid = AxesGrid(fig, 111, nrows_ncols=(1, 2),
            #                 axes_pad=0.7, cbar_mode='single', cbar_location='right',
            #                 cbar_pad=0.5)
            #
            # mats_to_plot = [frac_sub_eos, mean_max_eos]
            # titles = ['Fraction Subjects > Chance', 'Max Accuracy']
            # for i_ax, ax in enumerate(grid):
            #     im = ax.imshow(mats_to_plot[i_ax], interpolation='nearest', aspect='auto', vmin=chance,
            #                    vmax=1.0)
            #     ax.set_title(titles[i_ax])
            #     ax.set_xticks(range(len(num_insts)))
            #     ax.set_xticklabels(num_insts)
            #     ax.set_yticks(range(len(win_lens)))
            #     ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
            #     ax.text(-0.15, 1.05, string.ascii_uppercase[i_ax], transform=ax.transAxes,
            #             size=20, weight='bold')
            #     ax.set_xlabel('Number of Instances')
            #     if i_ax == 0:
            #         ax.set_ylabel('Window Length (ms)')
            # cbar = grid.cbar_axes[0].colorbar(im)
            # fig.suptitle('Accuracy and Consistency Scores\nDecoding {word} Post-Sentence'.format(
            #     word=PLOT_TITLE_WORD[word]),
            #              fontsize=18)
            # # fig.tight_layout()
            # fig.savefig(fig_fname.format(
            #     exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest, fig_type='single-mean-score-comp'
            # ), bbox_inches='tight')

            z_frac_eos = frac_sub_eos - 0.5
            z_frac_eos /= np.max(z_frac_eos)

            z_max_eos = z_max_eos - chance[word]
            z_max_eos /= np.max(z_max_eos)
            all_combined_z = (z_frac_eos + z_max_eos) / 2.0
            # print(np.any(np.isnan(all_combined)))
            # all_combined_z = zscore(all_combined)
            # print(np.any(np.isnan(all_combined_z)))
            # all_combined_z[np.isnan(all_combined_z)] = 0.0
            combo_scores.append(all_combined_z[None, ...])
            sen_combo_scores.append(all_combined_z[None, ...])

            im = combo_grid[i_word].imshow(all_combined_z, interpolation='nearest', aspect='auto', vmin=-3.0, vmax=3.0)
            combo_grid[i_word].set_title('Decoding {word}\nfrom {sen}'.format(sen=PLOT_TITLE_SEN[sen_type],
                                                                               word=PLOT_TITLE_WORD[word]))
            combo_grid[i_word].set_xticks(range(len(num_insts)))
            combo_grid[i_word].set_xticklabels(num_insts)
            combo_grid[i_word].set_yticks(range(len(win_lens)))
            combo_grid[i_word].set_yticklabels(np.array(win_lens).astype('float') * 2)
            combo_grid[i_word].text(-0.15, 1.05, string.ascii_uppercase[i_word], transform=combo_grid[i_word].transAxes,
                                     size=20, weight='bold')
            if i_word > 2:
                combo_grid[i_word].set_xlabel('Number of Instances')
            if i_word == 0 or i_word == 2 or i_word == 4:
                combo_grid[i_word].set_ylabel('Window Length (ms)')

        cbar = combo_grid.cbar_axes[0].colorbar(im)
        combo_fig.suptitle('Post-Sentence Combined Scores',
            fontsize=18)

        combo_fig.savefig(fig_fname.format(
            exp=args.experiment, sen_type=sen_type, word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            fig_type='word-sen-combo-max-score-comp'
        ), bbox_inches='tight')

        all_combined_sen = np.sum(np.concatenate(sen_combo_scores, axis=0), axis=0)
        optimal = np.unravel_index(np.argmax(all_combined_sen), all_combined_sen.shape)

        fig, ax = plt.subplots()
        h = ax.imshow(all_combined_sen, interpolation='nearest', vmin=-7.0, vmax=7.0)
        plt.colorbar(h)
        ax.set_title('{sen} Total Combined Score'.format(sen=PLOT_TITLE_SEN[sen_type]),
                     fontsize=14)
        ax.set_xticks(range(len(num_insts)))
        ax.set_xticklabels(num_insts)
        ax.set_yticks(range(len(win_lens)))
        ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
        ax.set_xlabel('Number of Instances')
        ax.set_ylabel('Window Length (ms)')
        # fig.tight_layout()
        plt.savefig(fig_fname.format(
            exp=args.experiment, sen_type=sen_type, word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            fig_type='total-comb-max-score-comp'
        ), bbox_inches='tight')

        print(
            'Optimal window size: {win}\nOptimal number of instances: {ni}\nScore: {score}'.format(
                win=win_lens[optimal[0]],
                ni=num_insts[optimal[1]],
                score=np.max(
                    all_combined_sen)))

    all_combined = np.sum(np.concatenate(combo_scores, axis=0), axis=0)
    optimal = np.unravel_index(np.argmax(all_combined), all_combined.shape)

    fig, ax = plt.subplots()
    h = ax.imshow(all_combined, interpolation='nearest', vmin=-7.0, vmax=7.0)
    plt.colorbar(h)
    ax.set_title('Post-Sentence Total Combined Score',
        fontsize=14)
    ax.set_xticks(range(len(num_insts)))
    ax.set_xticklabels(num_insts)
    ax.set_yticks(range(len(win_lens)))
    ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
    ax.set_xlabel('Number of Instances')
    ax.set_ylabel('Window Length (ms)')
    # fig.tight_layout()
    plt.savefig(fig_fname.format(
        exp=args.experiment, sen_type='all', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
        fig_type='total-comb-max-score-comp'
    ), bbox_inches='tight')

    print(
        'Optimal window size: {win}\nOptimal number of instances: {ni}\nScore: {score}'.format(win=win_lens[optimal[0]],
                                                                                               ni=num_insts[optimal[1]],
                                                                                               score=np.max(
                                                                                                   all_combined)))

    plt.show()