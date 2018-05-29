import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_alg_comp
from mpl_toolkits.axes_grid1 import AxesGrid
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
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice'}

ALG_LABELS = {'lr-l1': 'Logistic L1',
              'lr-l2': 'Logistic L2',
              'lr-None': 'Logistic',
              'gnb': 'GNB FS',
              'gnb-None': 'GNB',
              'svm-l1': 'SVM L1',
              'svm-l2': 'SVM L2'}

def intersect_accs(exp,
                   word,
                   alg,
                   win_len=100,
                   overlap=12,
                   adj=None,
                   num_instances=1,
                   avgTime='F',
                   avgTest='F'):
    top_dir = run_alg_comp.TOP_DIR.format(exp=exp)

    acc_by_sub = []
    acc_intersect = []
    time_by_sub = []
    win_starts_by_sub = []
    for sub in load_data.VALID_SUBS[exp]:
        save_dir = run_alg_comp.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result_fname = run_alg_comp.SAVE_FILE.format(dir=save_dir,
                                                     sub=sub,
                                                     word=word,
                                                     win_len=win_len,
                                                     ov=overlap,
                                                     perm='F',
                                                     alg=alg,
                                                     adj=adj,
                                                     avgTm=avgTime,
                                                     avgTst=avgTest,
                                                     inst=num_instances,
                                                     rsP=1) + '.npz'
        if not os.path.isfile(result_fname):
            continue
        try:
            result = np.load(result_fname)
            time = np.squeeze(result['time'])
            win_starts = result['win_starts']
            fold_acc = result['tgm_acc']
        except:
            print(result_fname)
            continue
        # print(result_fname)
        if 'gnb' in  alg:
            acc = fold_acc
        else:
            acc = np.mean(fold_acc, axis=0)
        time_by_sub.append(time[None, ...])
        win_starts_by_sub.append(win_starts[None, ...])
        acc_thresh = acc > 0.5
        acc_by_sub.append(acc[None, ...])
        acc_intersect.append(acc_thresh[None, ...])
    if len(acc_by_sub) == 0:
        return [], [], [], []
    acc_all = np.concatenate(acc_by_sub, axis=0)
    intersection = np.sum(np.concatenate(acc_intersect, axis=0), axis=0)
    time = np.mean(np.concatenate(time_by_sub, axis=0), axis=0)
    win_starts = np.mean(np.concatenate(win_starts_by_sub, axis=0), axis=0).astype('int')

    return intersection, acc_all, time, win_starts


if __name__ == '__main__':
    alg_list = run_alg_comp.VALID_ALGS
    win_list = [2, 12, 25, 50, 100]
    inst_list = [1, 2, 5, 10]
    avgTime_list = ['T', 'F']
    avgTest_list = ['T', 'F']

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', default='lr-l2', choices=run_alg_comp.VALID_ALGS)
    parser.add_argument('--win_len', type=int, default=2, choices=win_list)
    parser.add_argument('--num_instances', type=int, default=1, choices=inst_list)
    parser.add_argument('--avgTime', default='T', choices=avgTime_list)
    parser.add_argument('--avgTest', default='T', choices=avgTest_list)
    parser.add_argument('--overlap', type=int, default=2)
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    args = parser.parse_args()

    exp = 'krns2'
    word = 'voice'
    overlap = args.overlap
    global_alg = args.alg
    global_win = args.win_len
    global_inst = args.num_instances
    global_avgTime = args.avgTime
    global_avgTest = args.avgTest
    adj = args.adj


    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20


    fig, ax = plt.subplots(figsize=(10,10))
    colors = ['b', 'c', 'm', 'r', 'g', 'y', 'k']
    max_acc = np.zeros((len(alg_list),))
    max_std = np.zeros((len(alg_list),))
    for i_alg, alg in enumerate(alg_list):
        if 'l2' in alg:
            adj_alg='zscore'
        else:
            adj_alg=None
        intersection, acc_all, time, win_starts = intersect_accs(exp,
                                                                 word,
                                                                 alg,
                                                                 win_len=2,
                                                                 overlap=overlap,
                                                                 adj=adj_alg,
                                                                 num_instances=1,
                                                                 avgTime='T',
                                                                 avgTest='T')
        if len(intersection) > 0:
            print(acc_all.shape)
            diag_acc = np.diag(np.mean(acc_all, axis=0))
            diag_std = np.diag(np.std(acc_all, axis=0))
            num_sub = np.sqrt(float(acc_all.shape[0]))
            max_time = np.argmax(diag_acc)
            max_acc[i_alg] = diag_acc[max_time]
            max_std[i_alg] = diag_std[max_time]
            diag_std = np.divide(diag_std, num_sub)
            diag_time = time[win_starts] + global_win*0.002 - 0.5
            ax.plot(diag_time, diag_acc, color=colors[i_alg], label=ALG_LABELS[alg])
            ax.fill_between(diag_time, diag_acc - diag_std, diag_acc + diag_std, facecolor=colors[i_alg], edgecolor='w', alpha=0.3)
    ax.axhline(0.5, color='k', label='Chance')
    ax.set_xlabel('Time relative to Sentence Offset (s)', fontsize=axislabelsize)
    ax.set_ylabel('Classification Accuracy', fontsize=axislabelsize)
    ax.tick_params(labelsize=ticklabelsize)
    ax.legend(loc=3, fontsize=legendfontsize)
    fig.suptitle('Algorithm Comparison\nVoice Decoding Post-Sentence', fontsize=suptitlesize)
    fig.subplots_adjust(top=0.85)
    fig.savefig('/home/nrafidi/thesis_figs/alg_comp_over_time.pdf')
    fig.savefig('/home/nrafidi/thesis_figs/alg_comp_over_time.png')


    result = np.load('/share/volume0/nrafidi/alg_times_new.npz')
    alg_times = result['min_times']
    alg_names = result['algs'].tolist()
    alg_times_ordered = []
    for a in alg_list:
        alg_times_ordered.append(alg_times[alg_names.index(a)])
    alg_times_ordered = np.array(alg_times_ordered)
    alg_times_ordered /= np.max(alg_times_ordered)
    print(max_acc)
    bar_fig, bar_ax = plt.subplots(figsize=(10, 10))
    ind = np.arange(len(alg_list))
    width = 0.3
    bar_ax.bar(ind, max_acc, width, yerr=max_std, color='b', ecolor='r', label='Max Accuracy')
    bar_ax.bar(ind + width, alg_times_ordered, width, color='g', label='Runtime as fraction of max')
    bar_ax.axhline(0.5, color='k', label='Chance Accuracy')
    bar_ax.set_xticks(ind + width) #/ 2.0)
    bar_ax.set_xticklabels([ALG_LABELS[alg] for alg in alg_list]) #, fontdict={'horizontalalignment': 'center'})
    bar_ax.set_ylim([0.0, 1.0])
    bar_fig.suptitle('Algorithm\nMax Accuracy Comparison', fontsize=suptitlesize)
    bar_ax.tick_params(labelsize=ticklabelsize)
    bar_ax.legend(fontsize=legendfontsize)
    bar_ax.set_xlabel('Algorithm', fontsize=axislabelsize)
    bar_ax.set_ylabel('Classiciation Accuracy/Runtime Fraction', fontsize=axislabelsize)
    bar_fig.subplots_adjust(top=0.85)
    bar_fig.savefig('/home/nrafidi/thesis_figs/alg_comp_bar.pdf')
    bar_fig.savefig('/home/nrafidi/thesis_figs/alg_comp_bar.png')

    # adj = None
    win_fig = plt.figure(figsize=(20, 8))
    win_grid = AxesGrid(win_fig, 111, nrows_ncols=(1, 2),
                        axes_pad=0.7)
    max_acc = np.zeros((len(win_list),2))
    max_std = np.zeros((len(win_list),2))
    win_labels = []
    avg_labels = []
    for i_avg, avgTime in enumerate(avgTime_list):
        if avgTime == 'T':
            avg_time_str = 'Time Average'
        else:
            avg_time_str = 'No Time Average'
        avg_labels.append(avg_time_str)
        ax = win_grid[i_avg]
        for i_win, win in enumerate(win_list):
            intersection, acc_all, time, win_starts = intersect_accs(exp,
                                                                     word,
                                                                     global_alg,
                                                                     win_len=win,
                                                                     overlap=overlap,
                                                                     adj=adj,
                                                                     num_instances=global_inst,
                                                                     avgTime=avgTime,
                                                                     avgTest=global_avgTest)
            if type(acc_all) != list:
                print('{} {}'.format(avgTime, win))
            if len(intersection) > 0:
                diag_acc = np.diag(np.mean(acc_all, axis=0))
                diag_std = np.diag(np.std(acc_all, axis=0))
                num_sub = np.sqrt(float(acc_all.shape[0]))
                max_time = np.argmax(diag_acc)
                max_acc[i_win, i_avg] = diag_acc[max_time]
                max_std[i_win, i_avg] = diag_std[max_time]
                diag_std = np.divide(diag_std, num_sub)
                win_in_s = win * 0.002
                diag_time = time[win_starts] + win_in_s - 0.5
                label_str = '%.3f s' % win_in_s
                if i_avg == 0:
                    win_labels.append(label_str)
                ax.plot(diag_time, diag_acc, color=colors[i_win], label=label_str)
                ax.fill_between(diag_time, diag_acc - diag_std, diag_acc + diag_std, facecolor=colors[i_win],
                                edgecolor='w', alpha=0.3)
        ax.axhline(0.5, color='k', label='Chance')
        # ax.set_xlabel('Time relative to Sentence Offset (s)', fontsize=axislabelsize)
        if i_avg == 0:
            ax.set_ylabel('Classification Accuracy', fontsize=axislabelsize)
        ax.legend(loc=3, fontsize=legendfontsize)
        ax.set_title(avg_time_str, fontsize=axistitlesize)
        ax.set_xlim([0.0, 0.5])
        ax.tick_params(labelsize=ticklabelsize)
        ax.text(-0.15, 1.05, string.ascii_uppercase[i_avg], transform=ax.transAxes,
                                size=axislettersize, weight='bold')
    win_fig.text(0.5, 0.04, 'Time relative to Sentence Offset (s)', fontsize=axislabelsize, ha='center')
    win_fig.subplots_adjust(top=0.8)
    win_fig.suptitle('Window Length Comparison\nVoice Decoding Post-Sentence', fontsize=suptitlesize)
    win_fig.savefig('/home/nrafidi/thesis_figs/win_comp_over_time.pdf')
    win_fig.savefig('/home/nrafidi/thesis_figs/win_comp_over_time.png')

    win_tgm_fig = plt.figure(figsize=(20, 8))
    win_tgm_grid = AxesGrid(win_tgm_fig, 111, nrows_ncols=(1, 2),
                        axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True)
    win = 50
    time_step = int(50 / args.overlap)
    time_adjust = win * 0.002 * time_step
    win_in_s = win * 0.002
    for i_avg, avgTime in enumerate(avgTime_list):
        if avgTime == 'T':
            avg_time_str = 'Time Average'
        else:
            avg_time_str = 'No Time Average'
        ax = win_tgm_grid[i_avg]
        intersection, acc_all, time, win_starts = intersect_accs(exp,
                                                                 word,
                                                                 global_alg,
                                                                 win_len=win,
                                                                 overlap=overlap,
                                                                 adj=adj,
                                                                 num_instances=global_inst,
                                                                 avgTime=avgTime,
                                                                 avgTest=global_avgTest)

        tgm_acc = np.mean(acc_all, axis=0)
        diag_time = time[win_starts] + win_in_s - 0.5
        num_time = tgm_acc.shape[0]
        min_time = 0.0
        max_time = 0.1 * num_time / time_step
        label_time = np.arange(min_time, max_time, 0.1)
        print(label_time)
        ax.set_xticks(np.arange(0.0, float(num_time), float(time_step)) - time_adjust)
        ax.set_yticks(np.arange(0.0, num_time, time_step) - time_adjust)
        ax.set_xticklabels(label_time)
        ax.set_yticklabels(label_time)
        im = ax.imshow(tgm_acc, interpolation='nearest', vmin=0.5, vmax=0.75)
        ax.text(-0.15, 1.05, string.ascii_uppercase[i_avg], transform=ax.transAxes,
                size=axislettersize, weight='bold')
        ax.set_title(avg_time_str, fontsize=axistitlesize)
        ax.tick_params(labelsize=ticklabelsize)
    label_str = '%.3f s' % win_in_s
    cbar = win_tgm_grid.cbar_axes[0].colorbar(im)
    win_tgm_fig.text(0.04, 0.15, 'Train Time Relative to Sentence Offset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    win_tgm_fig.text(0.5, 0.04, 'Test Time relative to Sentence Offset (s)', fontsize=axislabelsize, ha='center')
    win_tgm_fig.subplots_adjust(top=0.8)
    win_tgm_fig.suptitle('Averaging Comparison\nWindow Size: {}'.format(label_str), fontsize=suptitlesize)
    win_tgm_fig.savefig('/home/nrafidi/thesis_figs/win_tgm_comp.pdf')
    win_tgm_fig.savefig('/home/nrafidi/thesis_figs/win_tgm_comp.png')

    bar_fig, bar_ax = plt.subplots(figsize=(10, 10))
    ind = np.arange(len(win_list))
    width = 0.3
    bar_ax.bar(ind, max_acc[:, 0], width, yerr=max_std[:, 0], color='b', ecolor='r', label=avg_labels[0])
    bar_ax.bar(ind + width, max_acc[:, 1], width, yerr=max_std[:, 1], color='g', ecolor='r', label=avg_labels[1])
    bar_ax.set_xticks(ind + width) # / 2.0)
    bar_ax.set_xticklabels(win_labels)
    bar_ax.set_ylim([0.5, 1.0])
    bar_ax.tick_params(labelsize=ticklabelsize)
    bar_ax.legend(fontsize=legendfontsize)
    bar_ax.set_ylabel('Classification Accuracy', fontsize=axislabelsize)
    bar_ax.set_xlabel('Window Size (s)', fontsize=axislabelsize)
    bar_fig.suptitle('Window Size\nMax Accuracy Comparison', fontsize=suptitlesize)
    bar_fig.subplots_adjust(top=0.85)
    bar_fig.savefig('/home/nrafidi/thesis_figs/win_comp_bar.pdf')
    bar_fig.savefig('/home/nrafidi/thesis_figs/win_comp_bar.png')

    inst_fig = plt.figure(figsize=(20, 8))
    inst_grid = AxesGrid(inst_fig, 111, nrows_ncols=(1, 2),
                        axes_pad=0.7)
    max_acc = np.zeros((len(inst_list), 2))
    max_std = np.zeros((len(inst_list), 2))
    avg_labels = []
    for i_avg, avgTest in enumerate(avgTest_list):
        if avgTest == 'T':
            avg_test_str = 'Test Sample Average'
        else:
            avg_test_str = 'No Test Sample Average'
        avg_labels.append(avg_test_str)
        ax = inst_grid[i_avg]
        for i_inst, inst in enumerate(inst_list):
            intersection, acc_all, time, win_starts = intersect_accs(exp,
                                                                     word,
                                                                     global_alg,
                                                                     win_len=global_win,
                                                                     overlap=overlap,
                                                                     adj=adj,
                                                                     num_instances=inst,
                                                                     avgTime=global_avgTime,
                                                                     avgTest=avgTest)
            if type(acc_all) != list:
                print('{} {}'.format(avgTest, inst))
            if len(intersection) > 0:
                diag_acc = np.diag(np.mean(acc_all, axis=0))
                diag_std = np.diag(np.std(acc_all, axis=0))
                num_sub = np.sqrt(float(acc_all.shape[0]))
                max_time = np.argmax(diag_acc)
                max_acc[i_inst, i_avg] = diag_acc[max_time]
                max_std[i_inst, i_avg] = diag_std[max_time]
                diag_std = np.divide(diag_std, num_sub)
                win_in_s = global_win * 0.002
                diag_time = time[win_starts] + win_in_s - 0.5
                ax.plot(diag_time, diag_acc, color=colors[i_inst], label='{}'.format(inst))
                ax.fill_between(diag_time, diag_acc - diag_std, diag_acc + diag_std, facecolor=colors[i_inst],
                                edgecolor='w', alpha=0.3)
        ax.axhline(0.5, color='k', label='Chance')
        # ax.set_xlabel('Time relative to Sentence Offset (s)', fontsize=axislabelsize)
        ax.set_ylabel('Classification Accuracy', fontsize=axislabelsize)
        ax.legend(loc=3, fontsize=legendfontsize)
        ax.tick_params(labelsize=ticklabelsize)
        ax.set_title(avg_test_str, fontsize=axistitlesize)
        ax.set_xlim([0.0, 0.5])
        ax.text(-0.15, 1.05, string.ascii_uppercase[i_avg], transform=ax.transAxes,
                size=axislettersize, weight='bold')
    inst_fig.subplots_adjust(top=0.8)
    inst_fig.text(0.5, 0.04, 'Time relative to Sentence Offset (s)', fontsize=axislabelsize, ha='center')
    inst_fig.suptitle('Repetition Averaging Comparison\nVoice Decoding Post-Sentence', fontsize=suptitlesize)
    inst_fig.savefig('/home/nrafidi/thesis_figs/inst_comp_over_time.pdf')
    inst_fig.savefig('/home/nrafidi/thesis_figs/inst_comp_over_time.png')
    print(max_acc)
    bar_fig, bar_ax = plt.subplots(figsize=(10, 10))
    ind = np.arange(len(inst_list))
    width = 0.3
    bar_ax.bar(ind, max_acc[:, 0], width, yerr=max_std[:, 0], color='b', ecolor='r', label=avg_labels[0])
    bar_ax.bar(ind + width, max_acc[:, 1], width, yerr=max_std[:, 1], color='g', ecolor='r', label=avg_labels[1])
    bar_ax.set_xticks(ind + width) # / 2.0)
    bar_ax.set_xticklabels(inst_list)
    bar_ax.set_ylim([0.5, 1.0])
    bar_ax.set_xlabel('Number of Instances', fontsize=axislabelsize)
    bar_ax.set_ylabel('Classification Accuracy', fontsize=axislabelsize)
    bar_ax.legend(fontsize=legendfontsize)
    bar_ax.tick_params(labelsize=ticklabelsize)
    bar_fig.suptitle('Repetition Averaging\nMax Accuracy Comparison', fontsize=suptitlesize)
    bar_fig.subplots_adjust(top=0.85)
    bar_fig.savefig('/home/nrafidi/thesis_figs/inst_comp_bar.pdf')
    bar_fig.savefig('/home/nrafidi/thesis_figs/inst_comp_bar.png')

    plt.show()


    #
    # if args.avgTest == 'T':
    #     avg_test_str = 'Test Sample Average'
    # else:
    #     avg_test_str = 'No Test Sample Average'
    #
    # combo_fig = plt.figure(figsize=(30, 10))
    # combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, 3),
    #                       axes_pad=0.7, cbar_mode='single', cbar_location='right',
    #                       cbar_pad=0.5)
    # i_combo = 0
    # for i_sen, sen_type in enumerate(['active', 'passive']):
    #     for i_word, word in enumerate(['noun1', 'verb', 'noun2']):
    #
    #         intersection, acc_all, time, win_starts, eos_max = intersect_accs(args.experiment,
    #                                                                           sen_type,
    #                                                                           word,
    #                                                                           win_len=args.win_len,
    #                                                                           overlap=args.overlap,
    #                                                                           adj=args.adj,
    #                                                                           num_instances=args.num_instances,
    #                                                                           avgTime=args.avgTime,
    #                                                                           avgTest=args.avgTest)
    #
    #         frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
    #         mean_acc = np.mean(acc_all, axis=0)
    #         time_win = time[win_starts]
    #         num_time = len(time_win)
    #         print(sen_type)
    #         print(word)
    #         print(num_time)
    #         time_step = int(250/args.overlap)
    #         if sen_type == 'active':
    #             text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
    #             max_line = 2.51 * 2 * time_step
    #             start_line = time_step
    #             if word == 'noun1':
    #                 start_line -= 0.0
    #             elif word == 'verb':
    #                 end_point = num_time - 83
    #                 frac_sub = frac_sub[time_step:(time_step+end_point)]
    #                 mean_acc = mean_acc[time_step:(time_step+end_point), time_step:(time_step+end_point)]
    #                 time_win = time_win[time_step:(time_step+end_point)]
    #                 intersection = intersection[time_step:(time_step+end_point), time_step:(time_step+end_point)]
    #                 max_line -= 0.5
    #                 start_line -= 0.5
    #             else:
    #                 max_line -= 1.5
    #                 start_line -= 1.5
    #         else:
    #             text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
    #             max_line = 3.51 * 2 * time_step
    #             start_line = time_step
    #             if word == 'noun1':
    #                 start_line -= 0.0
    #             elif word == 'verb':
    #                 max_line -= 1.0
    #                 start_line -= 1.0
    #             else:
    #                 max_line -= 2.5
    #                 start_line -= 2.5
    #         print(mean_acc.shape)
    #         print(np.max(mean_acc))
    #
    #         ax = combo_grid[i_combo]
    #         im = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=0.25, vmax=0.5)
    #         if i_word == 0:
    #             ax.set_ylabel('Train Time (s)')
    #         if i_sen == 1:
    #             ax.set_xlabel('Test Time (s)')
    #         ax.set_title('{word} from {sen_type}'.format(
    #             sen_type=PLOT_TITLE_SEN[sen_type],
    #             word=PLOT_TITLE_WORD[word]), fontsize=14)
    #         ax.set_xticks(range(0, len(time_win), time_step))
    #         label_time = time_win
    #         label_time = label_time[::time_step]
    #         label_time[np.abs(label_time) < 1e-15] = 0.0
    #         ax.set_xticklabels(label_time)
    #         ax.set_yticks(range(0, len(time_win), time_step))
    #         ax.set_yticklabels(label_time)
    #         time_adjust = args.win_len
    #
    #         for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
    #             ax.axvline(x=v, color='w')
    #             if i_v < len(text_to_write):
    #                 ax.text(v + 0.05 * 2*time_step, 1.5*time_step, text_to_write[i_v], color='w', fontsize=10)
    #         ax.set_xlim(left=time_step)
    #         ax.set_ylim(top=time_step)

    #         i_combo += 1
    #
    #         fig, ax = plt.subplots()
    #         h = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=0.25, vmax=0.5)
    #         ax.set_ylabel('Train Time (s)')
    #         ax.set_xlabel('Test Time (s)')
    #         ax.set_title('Average TGM Decoding {word} from {sen_type}\n{avgTime}, {avgTest}\nNumber of Instances: {ni}'.format(
    #             sen_type=PLOT_TITLE_SEN[sen_type],
    #             word=PLOT_TITLE_WORD[word],
    #             avgTime=avg_time_str,
    #             avgTest=avg_test_str,
    #             ni=args.num_instances))
    #         ax.set_xticks(range(0, len(time_win), time_step))
    #         label_time = time_win
    #         label_time = label_time[::time_step]
    #         label_time[np.abs(label_time) < 1e-15] = 0.0
    #         ax.set_xticklabels(label_time)
    #         ax.set_yticks(range(0, len(time_win), time_step))
    #         ax.set_yticklabels(label_time)
    #         time_adjust = args.win_len
    #
    #         for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
    #             ax.axvline(x=v, color='w')
    #             if i_v < len(text_to_write):
    #                 plt.text(v + 0.05 * 2 * time_step, 1.5 * time_step, text_to_write[i_v], color='w')
    #         plt.colorbar(h)
    #         ax.set_xlim(left=time_step)
    #         ax.set_ylim(top=time_step)
    #         fig.tight_layout()
    #         plt.savefig(
    #             '/home/nrafidi/thesis_figs/{exp}_avg-tgm-title_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
    #                 exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
    #                 win_len=args.win_len,
    #                 overlap=args.overlap,
    #                 num_instances=args.num_instances
    #             ), bbox_inches='tight')
    #
    #
    #         fig, ax = plt.subplots()
    #         h = ax.imshow(np.squeeze(intersection), interpolation='nearest', aspect='auto', vmin=0, vmax=acc_all.shape[0])
    #
    #         ax.set_ylabel('Train Time (s)')
    #         ax.set_xlabel('Test Time (s)')
    #         ax.set_title(
    #             'Intersection TGM\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
    #                                                                       word=word,
    #                                                                       experiment=args.experiment))
    #         ax.set_xticks(range(0, len(time_win), time_step))
    #         label_time = time_win
    #         label_time = label_time[::time_step]
    #         label_time[np.abs(label_time) < 1e-15] = 0.0
    #         ax.set_xticklabels(label_time)
    #         ax.set_yticks(range(0, len(time_win), time_step))
    #         ax.set_yticklabels(label_time)
    #         ax.set_xlim(left=time_step)
    #         ax.set_ylim(top=time_step)
    #
    #         for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
    #             ax.axvline(x=v, color='k')
    #             if i_v < len(text_to_write):
    #                 plt.text(v + 0.05 * 2*time_step, time_step, text_to_write[i_v], color='k')
    #         plt.colorbar(h)
    #
    #         fig.tight_layout()
    #         plt.savefig(
    #             '/home/nrafidi/thesis_figs/{exp}_intersection_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #                 exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
    #                 win_len=args.win_len,
    #                 overlap=args.overlap,
    #                 num_instances=args.num_instances
    #             ), bbox_inches='tight')
    #
    #         time_adjust = args.win_len*0.002
    #         fig, ax = plt.subplots()
    #
    #         ax.plot(np.diag(mean_acc), label='Accuracy')
    #         ax.plot(frac_sub, label='Fraction of Subjects > Chance')
    #         # if sen_type == 'active':
    #         #     text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
    #         #     max_line = 2.0 - time_adjust + 0.01
    #         #     if word == 'noun1':
    #         #         start_line = -0.5 - time_adjust
    #         #     else:
    #         #         start_line = -1.0 - time_adjust
    #         #
    #         # else:
    #         #     text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
    #         #     max_line = 3.0 - time_adjust + 0.01
    #         #     if word == 'noun1':
    #         #         start_line = -0.5 - time_adjust
    #         #     else:
    #         #         start_line = -1.5 - time_adjust
    #
    #         ax.set_xticks(range(0, len(time_win), time_step))
    #         label_time = time_win
    #         label_time = label_time[::time_step]
    #         label_time[np.abs(label_time) < 1e-15] = 0.0
    #         ax.set_xticklabels(label_time)
    #         for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
    #             ax.axvline(x=v, color='k')
    #             if i_v < len(text_to_write):
    #                 plt.text(v + 0.05, 0.8, text_to_write[i_v])
    #         ax.set_ylabel('Accuracy/Fraction > Chance')
    #         ax.set_xlabel('Time')
    #         ax.set_ylim([0.0, 1.0])
    #         ax.set_xlim(left=time_step)
    #         ax.legend(loc=4)
    #         ax.set_title('Mean Acc over subjects and Frac > Chance\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
    #                                                                                      word=word,
    #                                                                                      experiment=args.experiment))
    #
    #         fig.tight_layout()
    #         plt.savefig(
    #             '/home/nrafidi/thesis_figs/{exp}_diag_acc_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #                 exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
    #                 win_len=args.win_len,
    #                 overlap=args.overlap,
    #                 num_instances=args.num_instances
    #             ), bbox_inches='tight')
    #
    # cbar = combo_grid.cbar_axes[0].colorbar(im)
    # combo_fig.suptitle('TGM Averaged Over Subjects',
    #     fontsize=18)
    #
    # combo_fig.savefig('/home/nrafidi/thesis_figs/{exp}_avg-tgm_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
    #                 exp=args.experiment, sen_type='both', word='all', avgTime=args.avgTime, avgTest=args.avgTest,
    #                 win_len=args.win_len,
    #                 overlap=args.overlap,
    #                 num_instances=args.num_instances
    #             ), bbox_inches='tight')
    #
    # plt.show()


