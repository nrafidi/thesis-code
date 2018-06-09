import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import run_TGM_LOSO
from mpl_toolkits.axes_grid1 import AxesGrid


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
    for sub in run_TGM_LOSO.VALID_SUBS[exp]:
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
    parser.add_argument('--num_sub', type=int)
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=10)
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
    legendfontsize = 14
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20
    num_plots = len(word_list)
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step

    num_sub = float(len(run_TGM_LOSO.VALID_SUBS[args.experiment]))

    num_sub_to_plot = args.num_sub

    sen_accs = []
    sen_stds = []
    sub_sen_diags = []
    sen_fracs = []
    sen_time = []

    sub_task_diags = []
    sub_sen_word_diags = []
    for i_sen, sen_type in enumerate(sen_type_list):
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

            num_sub = acc_all.shape[0]
            sub_diags = np.concatenate([np.diag(acc_all[i, :, :])[None, :] for i in range(num_sub)], axis=0)
            print(sub_diags.shape)
            sub_word_diags.append(sub_diags[None, :])
            if word == 'noun1':
                time = word_time
                win_starts = word_win_starts
        sen_time.append(time[win_starts])
        sub_word_diags = np.concatenate(sub_word_diags, axis=0)
        sub_sen_word_diags.append(sub_word_diags[None, ...])
        print(sub_word_diags.shape)
        sub_task_diags.append(sub_word_diags - chance[word])

    sub_sen_word_diags = np.concatenate(sub_sen_word_diags, axis=0)
    print(sub_sen_word_diags.shape)
    sub_task_diags = np.concatenate(sub_task_diags, axis=0)
    print(sub_task_diags.shape)

    num_time = len(win_starts)
    colors = ['r', 'b', 'g']
    xtick_array = np.arange(0, len(time[win_starts]), time_step) - time_adjust

    sub_scores = np.max(sub_task_diags, axis=2)
    print(sub_scores.shape)
    meow = np.max(sub_scores, axis=0)
    print(meow.shape)
    sub_scores /= meow[None, :]
    sub_scores = np.sum(sub_scores, axis=0)
    print(sub_scores.shape)

    sorted_subs = np.argsort(sub_scores)

    worst_subs = sorted_subs[:num_sub_to_plot]
    best_subs = sorted_subs[(num_sub-1):(num_sub - 1 - num_sub_to_plot):-1]

    best_sub_fig = plt.figure(figsize=(8*num_sub_to_plot, 8))
    best_sub_axs = AxesGrid(best_sub_fig, 111, nrows_ncols=(num_sub_to_plot, len(sen_type_list)),
                          axes_pad=0.2, share_all=True, direction='column')
    worst_sub_fig = plt.figure(figsize=(8 * num_sub_to_plot, 8))
    worst_sub_axs = AxesGrid(worst_sub_fig, 111, nrows_ncols=(num_sub_to_plot, len(sen_type_list)),
                            axes_pad=0.2, share_all=True, direction='column')

    act_sub_fig = plt.figure(figsize=(8 * num_sub_to_plot, 8))
    act_sub_axs = AxesGrid(act_sub_fig, 111, nrows_ncols=(num_sub_to_plot, 2),
                             axes_pad=0.2, share_all=True, direction='row')

    pass_sub_fig = plt.figure(figsize=(8 * num_sub_to_plot, 8))
    pass_sub_axs = AxesGrid(pass_sub_fig, 111, nrows_ncols=(num_sub_to_plot, 2),
                           axes_pad=0.2, share_all=True, direction='row')

    i_bw_fig = 0
    for i_sen, sen_type in enumerate(sen_type_list):
        if sen_type == 'active':
            text_to_write = ['Det', 'Noun', 'Verb', 'Det', 'Noun.']
            max_line = 2.51 * 2 * time_step - time_adjust
            start_line = time_step - time_adjust
            curr_fig = act_sub_fig
            curr_axs = act_sub_axs
        else:
            text_to_write = ['Det', 'Noun', 'was', 'Verb', 'by', 'Det', 'Noun.']
            max_line = 3.51 * 2 * time_step - time_adjust
            start_line = time_step - time_adjust
            curr_fig = pass_sub_fig
            curr_axs = pass_sub_axs
        i_ap_fig = 0
        for j_sub in range(num_sub_to_plot):
            best_sub_axs[i_bw_fig].set_xticks(xtick_array)
            worst_sub_axs[i_bw_fig].set_xticks(xtick_array)

            for i_ax in range(2):
                curr_axs[i_ap_fig + i_ax].set_xticks(xtick_array)

            for k_word, word in enumerate(word_list):
                color = colors[k_word]

                good_sub = sub_sen_word_diags[i_sen, k_word, best_subs[j_sub], :]
                bad_sub = sub_sen_word_diags[i_sen, k_word, worst_subs[j_sub], :]

                if sen_type == 'active' and args.alg == 'lr-l2':
                    if word == 'verb':
                        good_sub = good_sub[time_step:]
                        bad_sub = bad_sub[time_step:]
                    elif word == 'noun2':
                        good_sub = good_sub[2*time_step:]
                        bad_sub = bad_sub[2*time_step:]

                best_sub_axs[i_bw_fig].plot(range(len(good_sub)), good_sub,
                                                label='{word}'.format(word=word),
                                                color=color)
                curr_axs[i_ap_fig].plot(range(len(good_sub)), good_sub,
                                                label='{word}'.format(word=word),
                                                color=color)
                worst_sub_axs[i_bw_fig].plot(range(len(bad_sub)), bad_sub,
                                                label='{word}'.format(word=word),
                                                color=color)
                curr_axs[i_ap_fig + 1].plot(range(len(bad_sub)), bad_sub,
                                                 label='{word}'.format(word=word),
                                                 color=color)

            min_time = -0.5
            max_time = 0.5*len(time[win_starts])/time_step
            label_time = np.arange(min_time, max_time, 0.5)
            best_sub_axs[i_bw_fig].axhline(y=0.25, color='k', linestyle='dashed', label='chance')
            worst_sub_axs[i_bw_fig].axhline(y=0.25, color='k', linestyle='dashed', label='chance')
            best_sub_axs[i_bw_fig].set_xticklabels(label_time)
            worst_sub_axs[i_bw_fig].set_xticklabels(label_time)
            best_sub_axs[i_bw_fig].tick_params(labelsize=ticklabelsize)
            worst_sub_axs[i_bw_fig].tick_params(labelsize=ticklabelsize)
            for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
                best_sub_axs[i_bw_fig].axvline(x=v, color='k')
                worst_sub_axs[i_bw_fig].axvline(x=v, color='k')
                if i_v < len(text_to_write):
                    best_sub_axs[i_bw_fig].text(v + 0.15, 0.9, text_to_write[i_v])
                    worst_sub_axs[i_bw_fig].text(v + 0.15, 0.9, text_to_write[i_v])

            best_sub_axs[i_bw_fig].set_ylim([0.0, 1.0])
            worst_sub_axs[i_bw_fig].set_ylim([0.0, 1.0])
            best_sub_axs[i_bw_fig].set_xlim([start_line, max_line + time_step*5])
            worst_sub_axs[i_bw_fig].set_xlim([start_line, max_line + time_step * 5])

            for i_ax in range(2):
                curr_axs[i_ap_fig + i_ax].axhline(y=0.25, color='k', linestyle='dashed', label='chance accuracy')
                curr_axs[i_ap_fig + i_ax].set_xticklabels(label_time)
                curr_axs[i_ap_fig + i_ax].tick_params(labelsize=ticklabelsize)
                for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
                    curr_axs[i_ap_fig + i_ax].axvline(x=v, color='k')
                    if i_v < len(text_to_write):
                        curr_axs[i_ap_fig + i_ax].text(v + 0.15, 0.9, text_to_write[i_v])
                curr_axs[i_ap_fig + i_ax].set_ylim([0.0, 1.0])
                curr_axs[i_ap_fig + i_ax].set_xlim([start_line, max_line + time_step * 5])

            curr_axs[i_ap_fig + 1].legend(loc=2, bbox_to_anchor=(0.65, 0.8), fontsize=legendfontsize)
            if i_sen == 1:
                best_sub_axs[i_bw_fig].legend(loc=2, bbox_to_anchor=(0.65, 0.8), fontsize=legendfontsize)
                worst_sub_axs[i_bw_fig].legend(loc=2, bbox_to_anchor=(0.65, 0.8), fontsize=legendfontsize)
            best_sub_axs[i_bw_fig].set_title('{sen_type}'.format(sen_type=PLOT_TITLE_SEN[sen_type]), fontsize=axistitlesize)
            worst_sub_axs[i_bw_fig].set_title('{sen_type}'.format(sen_type=PLOT_TITLE_SEN[sen_type]),
                                                 fontsize=axistitlesize)
            good_sub_name = run_TGM_LOSO.VALID_SUBS[args.experiment][best_subs[j_sub]]
            bad_sub_name = run_TGM_LOSO.VALID_SUBS[args.experiment][worst_subs[j_sub]]

            curr_axs[i_ap_fig].set_title('Good Subject: {}'.format(good_sub_name), fontsize=axistitlesize)
            curr_axs[i_ap_fig + 1].set_title('Bad Subject: {}'.format(bad_sub_name), fontsize=axistitlesize)

            if i_sen == 1:
                best_sub_axs[i_bw_fig].text(-0.125, 1.125, good_sub_name, transform=best_sub_axs[i_bw_fig].transAxes,
                        size=axislettersize, weight='bold')
                worst_sub_axs[i_bw_fig].text(-0.125, 1.125, bad_sub_name, transform=worst_sub_axs[i_bw_fig].transAxes,
                                                size=axislettersize, weight='bold')
            i_bw_fig += 1
            i_ap_fig += 2
        curr_fig.suptitle('Best/Worst {num} Subjects\n{sen_type} {experiment}'.format(num=num_sub_to_plot,
                                                                                      sen_type=sen_type,
                                                                                      experiment=args.experiment),
                          fontsize=suptitlesize)
        curr_fig.text(0.5, 0.04, 'Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
        curr_fig.text(0.04, 0.5, 'Accuracy', va='center',
                       rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
        # curr_fig.subplots_adjust(top=0.85)
        curr_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_diag_acc_top-bot{num}_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=args.experiment, num=num_sub_to_plot, sen_type=sen_type, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=args.win_len,
                overlap=args.overlap,
                num_instances=args.num_instances
            ), bbox_inches='tight')

    best_sub_fig.suptitle('Top {} Subjects'.format(num_sub_to_plot), fontsize=suptitlesize)
    worst_sub_fig.suptitle('Bottom {} Subjects'.format(num_sub_to_plot), fontsize=suptitlesize)



    best_sub_fig.text(0.5, 0.04, 'Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
    best_sub_fig.text(0.04, 0.5, 'Accuracy', va='center',
                  rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    worst_sub_fig.text(0.5, 0.04, 'Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
    worst_sub_fig.text(0.04, 0.5, 'Accuracy', va='center',
                      rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    # best_sub_fig.subplots_adjust(top=0.85)
    # worst_sub_fig.subplots_adjust(top=0.85)

    best_sub_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_top{num}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            exp=args.experiment, num=num_sub_to_plot, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    worst_sub_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_bottom{num}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            exp=args.experiment, num=num_sub_to_plot, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')


    plt.show()


