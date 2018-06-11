import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO
import tgm_loso_acc_multisub
import string


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                    'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                   'noun2': 'Second Noun',
                  'verb': 'Verb',
                  'voice': 'Voice'}

def bhy_multiple_comparisons_procedure(uncorrected_pvalues, alpha=0.05, assume_independence=False):
    # Benjamini-Hochberg-Yekutieli
    # originally from Mariya Toneva
    if len(uncorrected_pvalues.shape) == 1:
        uncorrected_pvalues = np.reshape(uncorrected_pvalues, (1, -1))

    # get ranks of all p-values in ascending order
    sorting_inds = np.argsort(uncorrected_pvalues, axis=1)
    ranks = sorting_inds + 1  # add 1 to make the ranks start at 1 instead of 0

    # calculate critical values under arbitrary dependence
    if assume_independence:
        dependency_constant = 1.0
    else:
        dependency_constant = np.sum(1.0 / ranks)
    critical_values = ranks * alpha / float(uncorrected_pvalues.shape[1] * dependency_constant)

    # find largest pvalue that is <= than its critical value
    sorted_pvalues = np.empty(uncorrected_pvalues.shape)
    sorted_critical_values = np.empty(critical_values.shape)
    for i in range(uncorrected_pvalues.shape[0]):
        sorted_pvalues[i, :] = uncorrected_pvalues[i, sorting_inds[i, :]]
        sorted_critical_values[i, :] = critical_values[i, sorting_inds[i, :]]
    bh_thresh = np.zeros((sorted_pvalues.shape[0],))
    for j in range(sorted_pvalues.shape[0]):
        for i in range(sorted_pvalues.shape[1] - 1, -1, -1):  # start from the back
            if sorted_pvalues[j, i] <= sorted_critical_values[j, i]:
                bh_thresh[j] = sorted_pvalues[j, i]
                print('threshold for row {} is: {}; critical value: {} (i: {})'.format(
                    j, bh_thresh[j], sorted_critical_values[j, i], i))
                break
    return bh_thresh


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

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    sen_type_list = ['active', 'passive']
    if args.experiment == 'PassAct3':
        word_list = ['noun1', 'verb', 'noun2']
        chance = {'noun1': 0.25,
                  'verb': 0.25,
                  'noun2': 0.25}
    else:
        word_list = ['noun1', 'verb', 'noun2']
        chance = {'noun1': 0.25,
                  'verb': 0.25,
                  'noun2': 0.25}

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20
    num_plots = len(word_list)
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step



    sen_accs = []
    sen_stds = []
    sub_sen_diags = []
    sen_fracs = []
    sen_time = []

    sen_fig, sen_axs = plt.subplots(1, len(sen_type_list), figsize=(16, 8))
    for i_sen, sen_type in enumerate(sen_type_list):
        acc_diags = []
        std_diags = []
        frac_diags = []
        time = []
        win_starts = []
        sub_word_diags = []
        for word in word_list:
            top_dir = tgm_loso_acc_multisub.TOP_DIR.format(exp=args.experiment)
            multi_file = tgm_loso_acc_multisub.MULTI_SAVE_FILE.format(dir=top_dir,
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
                                                mode='acc')

            result = np.load(multi_file + '.npz')
            acc_all = result['tgm_acc']
            word_time = result['time']
            word_win_starts = result['win_starts']
            acc_diags.append(np.diag(np.mean(acc_all, axis=0)))

            if word == 'noun1':
                time = word_time
                win_starts = word_win_starts
        sen_accs.append(acc_diags)
        sen_time.append(time[win_starts])
        num_time = len(win_starts)
        if sen_type == 'active':
            text_to_write = ['Det', 'Noun', 'Verb', 'Det', 'Noun.']
            max_line = 2.51 * 2 * time_step - time_adjust
            start_line = 0.0 - time_adjust
        else:
            text_to_write = ['Det', 'Noun', 'was', 'Verb', 'by', 'Det', 'Noun.']
            max_line = 3.51 * 2 * time_step - time_adjust
            start_line = 0.0 - time_adjust


        colors = ['r', 'b', 'g']
        ax = sen_axs[i_sen]
        xtick_array = np.arange(0, len(time[win_starts]), time_step) - time_adjust
        ax.set_xticks(xtick_array)
        print(xtick_array.shape)
        for i_word, word in enumerate(word_list):
            color = colors[i_word]
            acc = acc_diags[i_word]

            ax.plot(range(len(acc)), acc, label='{word} accuracy'.format(word=word), color=color)

            num_time = len(acc)

            # pval_thresh = bhy_multiple_comparisons_procedure(pvals, assume_independence=args.indep)
            #
            # for i_pt in range(num_time):
            #     if pvals[i_pt] <= pval_thresh:
            #         ax.scatter(i_pt, 0.82 + 0.02*i_word, color=color, marker='*')

        min_time = -0.5
        max_time = 0.5*len(time[win_starts])/time_step
        label_time = np.arange(min_time, max_time, 0.5)
        ax.axhline(y=0.25, color='k', linestyle='dashed', label='chance accuracy')
        ax.set_xticklabels(label_time)
        ax.tick_params(labelsize=ticklabelsize)
        for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
            ax.axvline(x=v, color='k')
            if i_v < len(text_to_write):
                ax.text(v + 0.15, 0.9, text_to_write[i_v])
        if i_sen == 0:
            ax.set_ylabel('Accuracy', fontsize=axislabelsize)
        # ax.set_xlabel('Time Relative to Sentence Onset (s)')
        ax.set_ylim([0.0, 1.2])
        ax.set_xlim([start_line, max_line + time_step*5])
        if i_sen == 1:
            ax.legend(loc=2, bbox_to_anchor=(0.65, 0.8), fontsize=legendfontsize)
        ax.set_title('{sen_type}'.format(sen_type=PLOT_TITLE_SEN[sen_type]), fontsize=axistitlesize)
        ax.text(-0.05, 1.05, string.ascii_uppercase[i_sen], transform=ax.transAxes,
                size=axislettersize, weight='bold')

    sen_fig.suptitle('Accuracy', fontsize=suptitlesize)
    # sen_fig.tight_layout()
    sen_fig.text(0.5, 0.04, 'Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
    plt.subplots_adjust(top=0.85)
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            exp=args.experiment, sen_type='both', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type='both', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')


    plt.show()


