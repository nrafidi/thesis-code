import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import run_TGM_LOSO_EOS
import tgm_loso_acc_eos
from scipy.stats import wilcoxon

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice',
                   'propid': 'Proposition ID',
                   'senlen': 'Sentence Length'}

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

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
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='T')
    parser.add_argument('--sig_test', default='binomial', choices=['binomial', 'wilcoxon'])
    parser.add_argument('--indep', action='store_true')
    args = parser.parse_args()

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    sen_type = 'pooled'
    if args.experiment == 'krns2':
        word_list = ['voice', 'verb', 'agent', 'patient', 'noun1']
        if args.num_instances > 1:
            word_list.append('propid')
        chance = {'noun1': 0.125,
                  'verb': 0.25,
                  'agent': 0.25,
                  'patient': 0.25,
                  'voice': 0.5,
                  'propid': 1.0/16.0}
    else:
        word_list = ['voice', 'senlen', 'verb', 'noun1']

        chance = {'noun1': 0.25,
                  'verb': 0.25,
                  'agent': 0.25,
                  'patient': 0.25,
                  'voice': 0.5,
                  'propid': 1.0 / 24.0,
                  'senlen': 0.5}

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002

    sen_fig, ax = plt.subplots(figsize=(15, 12))
    acc_diags = []
    frac_diags = []
    time = []
    win_starts = []
    sub_word_diags = []
    for word in word_list:
        intersection, acc_all, word_time, word_win_starts, eos_max = tgm_loso_acc_eos.intersect_accs(args.experiment,
                                                                                                    sen_type,
                                                                                                    word,
                                                                                                    win_len=args.win_len,
                                                                                                    overlap=args.overlap,
                                                                                                    alg=args.alg,
                                                                                                    adj=args.adj,
                                                                                                    num_instances=args.num_instances,
                                                                                                    avgTime=args.avgTime,
                                                                                                    avgTest=args.avgTest)

        frac_diags.append(np.diag(intersection).astype('float')/float(acc_all.shape[0]))
        acc_diags.append(np.diag(np.mean(acc_all, axis=0)))
        num_sub = acc_all.shape[0]
        sub_diags = np.concatenate([np.diag(acc_all[i, :, :])[None, :] for i in range(num_sub)], axis=0)
        sub_word_diags.append(sub_diags[None, :])



        if word == 'voice':
            time = word_time
            win_starts = word_win_starts

    sub_word_diags = np.concatenate(sub_word_diags, axis=0)
    num_time = len(win_starts)
    max_line = 0.3 * 2 * time_step
    colors = ['r', 'g', 'b', 'm', 'c', 'k']

    for i_word, word in enumerate(word_list):
        color = colors[i_word]
        acc = acc_diags[i_word]
        frac = frac_diags[i_word]

        ax.plot(acc, label='{word} accuracy'.format(word=PLOT_TITLE_WORD[word]), color=color)
        pvals = np.empty((num_time,))
        for i_pt in range(num_time):
            if args.sig_test == 'binomial':
                num_above_chance = np.sum(np.squeeze(sub_word_diags[i_word, :, i_pt]) > chance[word])
                pvals[i_pt] = 0.5**num_above_chance
            else:
                _, pvals[i_pt] = wilcoxon(np.squeeze(sub_word_diags[i_word, :, i_pt]) - chance[word])
                if acc[i_pt] > chance[word]:
                    # print('meow')
                    pvals[i_pt] /= 2.0
                else:
                    # print('woof')
                    pvals[i_pt] = 1.0 - pvals[i_pt] / 2.0
        pval_thresh = bhy_multiple_comparisons_procedure(pvals, assume_independence=args.indep)
        for i_pt in range(num_time):
            if  pvals[i_pt]  <= pval_thresh:
                ax.scatter(i_pt, 0.88 - float(i_word)*0.02, color=color, marker='*')

    ax.set_xticks(range(0, len(time[win_starts]), time_step))
    label_time = time[win_starts]
    label_time = label_time[::time_step]
    label_time[np.abs(label_time) < 1e-15] = 0.0
    if args.experiment == 'krns2':
        ax.axhline(y=0.125, color='k', linestyle='dashdot', label='chance, noun1')
    ax.axhline(y=0.25, color='k', linestyle='dashed', label='chance, words')
    if args.experiment == 'PassAct3':
        ax.axhline(y=0.5, color='k', linestyle='dashdot', label='chance, voice/length')
    else:
        ax.axhline(y=0.5, color='k', linestyle='dashdot', label='chance, voice')
    if 'propid' in word_list:
        ax.axhline(y=1.0/16.0, color='k', linestyle=':', label='chance, proposition')
    ax.set_xticklabels(label_time)
    ax.axvline(x=max_line, color='k')
    ax.set_ylabel('Accuracy', fontsize=axislabelsize)
    ax.set_xlabel('Time Relative to Last Word Onset (s)', fontsize=axislabelsize)
    ax.set_ylim([0.0, 0.9])
    ax.set_xlim([0, len(time[win_starts]) + 0.8*time_step])
    ax.tick_params(labelsize=ticklabelsize)
    ax.legend(bbox_to_anchor=(0.75, 1.0), loc=2, borderaxespad=0., ncol=1, fontsize=legendfontsize)

    sen_fig.suptitle('Mean Accuracy over Subjects\nPost-Sentence', fontsize=suptitlesize)
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}_{sig}{indep}.pdf'.format(
            exp=args.experiment, sen_type=sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances,
            sig=args.sig_test,
            indep=args.indep
        ), bbox_inches='tight')
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}_{sig}{indep}.png'.format(
            exp=args.experiment, sen_type=sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances,
            sig=args.sig_test,
            indep=args.indep
        ), bbox_inches='tight')

    plt.show()


