import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO
import string
from scipy.stats import wilcoxon


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

def bhy_multiple_comparisons_procedure(uncorrected_pvalues, alpha=0.01, assume_independence=True):
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


def intersect_accs(exp,
                   sen_type,
                   word,
                   win_len=100,
                   overlap=12,
                   alg='lr-l2',
                   adj=None,
                   num_instances=1,
                   avgTime='F',
                   avgTest='F'):
    top_dir = run_TGM_LOSO.TOP_DIR.format(exp=exp)

    if sen_type == 'active':
        max_time = 2.0
    else:
        max_time = 3.0
    time_adjust = win_len * 0.002

    if num_instances == 1 and alg == 'lr-l1':
        avgTest = 'F'
    if exp == 'krns2' and alg == 'lr-l1': # and not (sen_type == 'active' and word == 'verb'):
        rep = 10
    else:
        rep = None

    acc_by_sub = []
    acc_intersect = []
    time_by_sub = []
    win_starts_by_sub = []
    eos_max_by_sub = []
    for sub in load_data.VALID_SUBS[exp]:
        save_dir = run_TGM_LOSO.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result_fname = run_TGM_LOSO.SAVE_FILE.format(dir=save_dir,
                                                       sub=sub,
                                                       sen_type=sen_type,
                                                       word=word,
                                                       win_len=win_len,
                                                       ov=overlap,
                                                       perm='F',
                                                       alg=alg,
                                                       adj=adj,
                                                       avgTm=avgTime,
                                                       avgTst=avgTest,
                                                       inst=num_instances,
                                                       rep=rep,
                                                       rsP=1,
                                                       mode='acc') + '.npz'
        if not os.path.isfile(result_fname):
            print(result_fname)
            continue
        result = np.load(result_fname)
        time = np.squeeze(result['time'])
        win_starts = result['win_starts']

        time_ind = np.where(time[win_starts] >= (max_time - time_adjust))
        time_ind = time_ind[0]

        fold_acc = result['tgm_acc']
        eos_max_fold = []
        for i_fold in range(fold_acc.shape[0]):
            diag_acc = np.diag(np.squeeze(fold_acc[i_fold, :, :]))
            argo = np.argmax(diag_acc[time_ind])
            eos_max_fold.append(time_ind[argo])
        eos_max_fold = np.array(eos_max_fold)
        eos_max_by_sub.append(eos_max_fold[None, :])
        acc = np.mean(fold_acc, axis=0)
        time_by_sub.append(time[None, ...])
        win_starts_by_sub.append(win_starts[None, ...])
        acc_thresh = acc > 0.25
        acc_by_sub.append(acc[None, ...])
        acc_intersect.append(acc_thresh[None, ...])
    acc_all = np.concatenate(acc_by_sub, axis=0)
    intersection = np.sum(np.concatenate(acc_intersect, axis=0), axis=0)
    time = np.mean(np.concatenate(time_by_sub, axis=0), axis=0)
    win_starts = np.mean(np.concatenate(win_starts_by_sub, axis=0), axis=0).astype('int')
    eos_max = np.concatenate(eos_max_by_sub, axis=0)

    return intersection, acc_all, time, win_starts, eos_max


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

    sen_type_list = ['active', 'passive']
    if args.experiment == 'PassAct3':
        word_list = ['noun1', 'verb']
        chance = {'noun1': 0.25,
                  'verb': 0.25}
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

    num_sub = float(len(run_TGM_LOSO.VALID_SUBS[args.experiment]))

    sen_accs = []
    sub_sen_diags = []
    sen_fracs = []
    sen_time = []

    sen_fig, sen_axs = plt.subplots(1, len(sen_type_list), figsize=(16, 8))
    for i_sen, sen_type in enumerate(sen_type_list):
        acc_diags = []
        frac_diags = []
        time = []
        win_starts = []
        sub_word_diags = []
        for word in word_list:
            intersection, acc_all, word_time, word_win_starts, eos_max = intersect_accs(args.experiment,
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
            if word == 'noun1':
                time = word_time
                win_starts = word_win_starts
        sen_accs.append(acc_diags)
        sen_fracs.append(frac_diags)
        sen_time.append(time[win_starts])
        sub_word_diags = np.concatenate(sub_word_diags, axis=0)
        sub_sen_diags.append(sub_word_diags[None, ...])
        num_time = len(win_starts)
        if sen_type == 'active':
            text_to_write = ['Det', 'Noun', 'Verb', 'Det', 'Noun.']
            max_line = 2.51 * 2 * time_step - time_adjust
            start_line = time_step - time_adjust
        else:
            text_to_write = ['Det', 'Noun', 'was', 'Verb', 'by', 'Det', 'Noun.']
            max_line = 3.51 * 2 * time_step - time_adjust
            start_line = time_step - time_adjust


        colors = ['r', 'g', 'b']
        ax = sen_axs[i_sen]
        for i_word, word in enumerate(word_list):
            color = colors[i_word]
            acc = acc_diags[i_word]
            frac = frac_diags[i_word]

            if sen_type == 'active' and args.alg == 'lr-l2':
                if word == 'verb':
                    acc = acc[time_step:]
                    frac = frac[time_step:]
                elif word == 'noun2':
                    acc = acc[2*time_step:]
                    frac = frac[2*time_step:]

            ax.plot(acc, label='{word} accuracy'.format(word=word), color=color)
            num_time = len(acc)
            pvals = np.empty((num_time,))
            for i_pt in range(num_time):
                if args.sig_test == 'binomial':
                    num_above_chance = np.sum(np.squeeze(sub_word_diags[i_word, :, i_pt]) > chance[word])
                    pvals[i_pt] = 0.5 ** num_above_chance
                else:
                    _, pvals[i_pt] = wilcoxon(np.squeeze(sub_word_diags[i_word, :, i_pt]) - chance[word])
                    if acc[i_pt] > chance[word]:
                        # print('meow')
                        pvals[i_pt] /= 2.0
                    else:
                        # print('woof')
                        pvals[i_pt] = 1.0 - pvals[i_pt]/2.0
                    # print(pvals[i_pt])
            pval_thresh = bhy_multiple_comparisons_procedure(pvals, assume_independence=args.indep)
            print(pval_thresh)
            print(np.min(pvals))
            for i_pt in range(num_time):
                if pvals[i_pt] <= pval_thresh:
                    if word == 'verb' and sen_type == 'active':
                        plot_pt = i_pt - time_step
                    elif word == 'noun2' and sen_type == 'active':
                        plot_pt = i_pt - 2*time_step
                    else:
                        plot_pt = i_pt
                    ax.scatter(plot_pt, 0.7 + 0.02*i_word, color=color, marker='*')

        ax.set_xticks(np.arange(0, len(time[win_starts]), time_step) - time_adjust)
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
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([start_line, max_line + time_step*5])

        ax.legend(loc=2, bbox_to_anchor=(0.64, 1.02), fontsize=legendfontsize)
        ax.set_title('{sen_type}'.format(sen_type=PLOT_TITLE_SEN[sen_type]), fontsize=axistitlesize)
        ax.text(-0.05, 1.05, string.ascii_uppercase[i_sen], transform=ax.transAxes,
                size=axislettersize, weight='bold')

    sen_fig.suptitle('Mean Accuracy over Subjects', fontsize=suptitlesize)
    # sen_fig.tight_layout()
    sen_fig.text(0.5, 0.04, 'Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
    plt.subplots_adjust(top=0.85)
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}_{sig}{indep}.pdf'.format(
            exp=args.experiment, sen_type='both', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances,
            sig=args.sig_test,
            indep=args.indep
        ), bbox_inches='tight')
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}_{sig}{indep}.png'.format(
            exp=args.experiment, sen_type='both', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances,
            sig=args.sig_test,
            indep=args.indep
        ), bbox_inches='tight')

    sub_sen_diags = np.concatenate(sub_sen_diags, axis=0)
    print(sub_sen_diags.shape)
    word_fig, word_axs = plt.subplots(1, len(word_list), figsize=(40, 10))
    for i_word, word in enumerate(word_list):

        text_to_write = [['Det', 'Noun1', 'Verb', 'Det', 'Noun2.'],
                         ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']]
        max_line = np.array([2.51 * 2 * time_step, 3.51 * 2 * time_step]) - time_adjust
        start_line = np.array([time_step, time_step]) - time_adjust


        colors = ['r', 'g']
        ax = word_axs[i_word]
        for i_sen, sen_type in enumerate(sen_type_list):
            color = colors[i_sen]
            acc = sen_accs[i_sen][i_word]
            frac = sen_fracs[i_sen][i_word]
            ax.plot(acc, label='{sen} accuracy'.format(sen=sen_type), color=color)
            pvals = np.empty((num_time,))
            for i_pt in range(num_time):
                if args.sig_test == 'binomial':
                    num_above_chance = np.sum(np.squeeze(sub_sen_diags[i_sen, i_word, :, i_pt]) > chance[word])
                    pvals[i_pt] = 0.5 ** num_above_chance
                else:
                    _, pvals[i_pt] = wilcoxon(np.squeeze(sub_word_diags[i_word, :, i_pt]) - chance[word])
            pval_thresh = bhy_multiple_comparisons_procedure(pvals, assume_independence=args.indep)
            print(pval_thresh)
            print(np.min(pvals))
            for i_pt in range(num_time):
                if pvals[i_pt] <= pval_thresh:
                    ax.scatter(i_pt, 0.7 - i_sen*0.1 + 0.06, color=color, marker='*')
            for i_v, v in enumerate(np.arange(start_line[i_sen], max_line[i_sen], time_step)):
                ax.axvline(x=v, color='k')
                if i_v < len(text_to_write[i_sen]):
                    ax.text(v + 0.15, 0.7 - i_sen*0.1, text_to_write[i_sen][i_v], color=color)

        ax.set_xticks(np.arange(0, len(sen_time[-1]), time_step) - time_adjust)
        min_time = -0.5
        max_time = 0.5 * len(sen_time[-1]) / time_step
        label_time = np.arange(min_time, max_time, 0.5)
        ax.set_xticklabels(label_time)
        if i_word == 0:
            ax.set_ylabel('Accuracy')
        ax.set_xlabel('Time Relative to Sentence Onset (s)')
        ax.set_ylim([0.0, 0.9])
        ax.set_xlim([start_line[-1], max_line[-1] + time_step * 5])
        ax.axhline(y=0.25, color='k', linestyle='dashed', label='chance accuracy')
        ax.legend(loc=1)
        ax.text(-0.05, 1.05, string.ascii_uppercase[i_word], transform=ax.transAxes,
                size=20, weight='bold')
        ax.set_title('{word}'.format(word=PLOT_TITLE_WORD[word]))

    word_fig.suptitle('Mean Accuracy over Subjects', fontsize=18)
    word_fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    word_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}_{sig}{indep}.pdf'.format(
            exp=args.experiment, word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances,
            sig=args.sig_test,
            indep=args.indep
        ), bbox_inches='tight')


    plt.show()


