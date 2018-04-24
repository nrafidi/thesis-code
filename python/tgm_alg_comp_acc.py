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
              'gnb': 'Naive Bayes',
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', default='lr-l1', choices=run_alg_comp.VALID_ALGS)
    parser.add_argument('--overlap', type=int, default=2)
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    args = parser.parse_args()

    exp = 'krns2'
    word = 'voice'
    overlap = args.overlap
    adj = args.adj
    global_alg = args.alg


    alg_list = run_alg_comp.VALID_ALGS
    win_list = [2, 12, 25, 50, 100]
    inst_list = [1, 2, 5, 10]
    avgTime_list = ['T', 'F']
    avgTest_list = ['T', 'F']


    fig, ax = plt.subplots()
    colors = ['r', 'b', 'g', 'c', 'm']
    for i_alg, alg in enumerate(alg_list):
        intersection, acc_all, time, win_starts = intersect_accs(exp,
                                                                 word,
                                                                 alg,
                                                                 win_len=win_list[0],
                                                                 overlap=overlap,
                                                                 adj=adj,
                                                                 num_instances=inst_list[0],
                                                                 avgTime=avgTime_list[0],
                                                                 avgTest=avgTest_list[0])
        if len(intersection) > 0:
            diag_acc = np.diag(np.mean(acc_all, axis=0))
            print(np.max(diag_acc))
            diag_time = time[win_starts] + win_list[0]*0.002 - 0.5
            ax.plot(diag_time, diag_acc, color=colors[i_alg], label=ALG_LABELS[alg])
    ax.axhline(0.5, color='k', label='Chance')
    ax.set_xlabel('Time relative to Sentence Offset (s)')
    ax.set_ylabel('Classification Accuracy')
    ax.legend()
    ax.set_title('Algorithm Comparison\nVoice Decoding Post-Sentence')

    plt.show()

    # if args.avgTime == 'T':
    #     avg_time_str = 'Time Average'
    # else:
    #     avg_time_str = 'No Time Average'
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
    #         ax.text(-0.15, 1.05, string.ascii_uppercase[i_combo], transform=ax.transAxes,
    #                                 size=20, weight='bold')
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


