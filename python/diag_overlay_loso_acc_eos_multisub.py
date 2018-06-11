import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import string
import tgm_loso_acc_eos_multisub

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice',
                   'propid': 'Proposition ID',
                   'senlen': 'Sentence Length'}

PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

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
    parser.add_argument('--win_len', type=int, default=25)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
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

    win_len = args.win_len
    num_instances = args.num_instances

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20
    time_step = int(250 / args.overlap)


    sen_type_list = ['active', 'passive', 'pooled']
    sen_fig, sen_axs = plt.subplots(1, len(sen_type_list), figsize=(36, 12))
    for i_sen_type, sen_type in enumerate(sen_type_list):
        time_adjust = win_len * 0.002
        if args.experiment == 'krns2':
            word_list = ['verb', 'agent', 'patient']
            if num_instances > 1:
                word_list.append('propid')
            if sen_type == 'pooled':
                word_list.append('noun1')
                word_list.append('voice')
            chance = {'noun1': 0.125,
                      'verb': 0.25,
                      'agent': 0.25,
                      'patient': 0.25,
                      'voice': 0.5,
                      'propid': 1.0/16.0}
        else:
            word_list = ['verb', 'agent', 'patient']
            if num_instances > 1:
                word_list.append('propid')
            if sen_type == 'pooled':
                word_list.append('voice')
                word_list.append('senlen')
            chance = {'noun1': 0.25,
                      'verb': 0.25,
                      'agent': 0.25,
                      'patient': 0.25,
                      'voice': 0.5,
                      'propid': 1.0 / 8.0,
                      'senlen': 0.5}



        # sen_fig, ax = plt.subplots(figsize=(15, 12))
        ax = sen_axs[i_sen_type]
        acc_diags = []
        time = []
        win_starts = []
        for i_word, word in enumerate(word_list):
            top_dir = tgm_loso_acc_eos_multisub.TOP_DIR.format(exp=args.experiment)
            multi_file = tgm_loso_acc_eos_multisub.MULTI_SAVE_FILE.format(dir=top_dir,
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

            if i_word == 0:
                time = word_time
                win_starts = word_win_starts

        num_time = len(win_starts)
        max_line = 0.3 * 2 * time_step
        colors = ['b', 'm', 'g', 'k', 'r', 'c']

        for i_word, word in enumerate(word_list):
            color = colors[i_word]
            acc = acc_diags[i_word]

            ax.plot(acc, label='{word}'.format(word=PLOT_TITLE_WORD[word]), color=color)


            # pval_thresh = bhy_multiple_comparisons_procedure(pvals, alpha=0.05, assume_independence=args.indep)
            # for i_pt in range(num_time):
            #     if  pvals[i_pt]  <= pval_thresh:
            #         ax.scatter(i_pt, 0.98 - float(i_word)*0.02, color=color, marker='*')
        ax.set_xticks(range(0, len(time[win_starts]), time_step))
        label_time = time[win_starts]
        label_time = label_time[::time_step]
        label_time[np.abs(label_time) < 1e-15] = 0.0

        ax.axhline(y=chance['agent'], color='k', linestyle='dashed')
        if sen_type == 'pooled':
            if args.experiment == 'krns2':
                ax.axhline(y=chance['noun1'], color='k', linestyle='dashdot')
            if args.experiment == 'PassAct3':
                ax.axhline(y=chance['senlen'], color='k', linestyle='dashdot')
            else:
                ax.axhline(y=chance['voice'], color='k', linestyle='dashdot')
            ax.axhline(y=chance['propid'], color='k', linestyle=':')
        ax.set_xticklabels(label_time)
        ax.axvline(x=max_line, color='k')
        if i_sen_type == 0:
            ax.set_ylabel('Accuracy', fontsize=axislabelsize)
        if i_sen_type == 1:
            ax.set_xlabel('Time Relative to Last Word Onset (s)', fontsize=axislabelsize)
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([0, len(time[win_starts]) + 0.8*time_step])
        ax.tick_params(labelsize=ticklabelsize)
        if sen_type == 'pooled':
            ax.legend(bbox_to_anchor=(1.5, 1.0), loc=1, borderaxespad=0., ncol=1, fontsize=legendfontsize)
        ax.set_title('{sen_type}'.format(sen_type=PLOT_TITLE_SEN[sen_type]), fontsize=axistitlesize)
        ax.text(-0.05, 1.05, string.ascii_uppercase[i_sen_type], transform=ax.transAxes,
                size=axislettersize, weight='bold')

    sen_fig.subplots_adjust(top=0.85)
    sen_fig.suptitle('Mean Accuracy over Subjects\nPost-Sentence', fontsize=suptitlesize)
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            exp=args.experiment, sen_type='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=win_len,
            overlap=args.overlap,
            num_instances=num_instances
        ), bbox_inches='tight')
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=win_len,
            overlap=args.overlap,
            num_instances=num_instances
        ), bbox_inches='tight')

    plt.show()


