import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO
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
    if exp == 'krns2' and not (sen_type == 'active' and word == 'verb') and alg == 'lr-l1':
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
        try:
            result = np.load(result_fname)
            time = np.squeeze(result['time'])
            win_starts = result['win_starts']
        except:
            print(result_fname)
            continue

        time_ind = np.where(time[win_starts] >= (max_time - time_adjust))
        time_ind = time_ind[0]

        fold_acc = result['tgm_acc']
        eos_max_fold = []
        for i_fold in range(fold_acc.shape[0]):
            diag_acc = np.diag(np.squeeze(fold_acc[i_fold, :, :]))
            # if sub =='C':
            #     fig, ax = plt.subplots()
            #     ax.imshow(np.squeeze(fold_acc[i_fold, :, :]), interpolation='nearest', aspect='auto')
            #     ax.set_title('{subject} {fold}'.format(subject=sub, fold=i_fold))
            argo = np.argmax(diag_acc[time_ind])
            eos_max_fold.append(time_ind[argo])
        eos_max_fold = np.array(eos_max_fold)
        eos_max_by_sub.append(eos_max_fold[None, :])
        acc = np.mean(fold_acc, axis=0)
        # if sub == 'B':
        #     fig, ax = plt.subplots()
        #     ax.plot(acc[:, 0])
        #     ax.set_title('B')

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

    if args.avgTime == 'T':
        avg_time_str = 'Time Average'
    else:
        avg_time_str = 'No Time Average'

    if args.avgTest == 'T':
        avg_test_str = 'Test Sample Average'
    else:
        avg_test_str = 'No Test Sample Average'

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20


    if args.experiment == 'krns2':
        word_list = ['noun1', 'verb', 'noun2']
    else:
        word_list = ['noun1', 'verb', 'noun2']
    num_plots = len(word_list)
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(6*num_plots, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, num_plots),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True)
    i_combo = 0
    for i_sen, sen_type in enumerate(['active', 'passive']):
        for i_word, word in enumerate(word_list):

            intersection, acc_all, time, win_starts, eos_max = intersect_accs(args.experiment,
                                                                              sen_type,
                                                                              word,
                                                                              win_len=args.win_len,
                                                                              overlap=args.overlap,
                                                                              alg=args.alg,
                                                                              adj=args.adj,
                                                                              num_instances=args.num_instances,
                                                                              avgTime=args.avgTime,
                                                                              avgTest=args.avgTest)
            frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
            mean_acc = np.mean(acc_all, axis=0)
            time_win = time[win_starts]

            if sen_type == 'active':
                text_to_write = ['Det', 'Noun', 'Verb', 'Det', 'Noun.']
                max_line = 2.51 * 2 * time_step - time_adjust
                start_line = - time_adjust
                if args.alg == 'lr-l2':
                    if word == 'noun1':
                        time_select = np.logical_and(time_win >= -0.5, time_win <= 4.0)
                    elif word == 'verb':
                        time_select = np.logical_and(time_win >= -1.0, time_win <= 3.5)
                    elif word == 'noun2':
                        time_select = np.logical_and(time_win >= -2.0, time_win <= 2.5)
                        # print(len(time_select))
                else:
                    if word == 'noun1':
                        start_line -= 0.0
                    elif word == 'verb':
                        end_point = num_time - 83
                        frac_sub = frac_sub[time_step:(time_step+end_point)]
                        mean_acc = mean_acc[time_step:(time_step+end_point), time_step:(time_step+end_point)]
                        time_win = time_win[time_step:(time_step+end_point)]
                        intersection = intersection[time_step:(time_step+end_point), time_step:(time_step+end_point)]
                        max_line -= 0.5
                        start_line -= 0.5
                    else:
                        max_line -= 1.5
                        start_line -= 1.5
            else:
                text_to_write = ['Det', 'Noun', 'was', 'Verb', 'by', 'Det', 'Noun.']
                max_line = 3.51 * 2 * time_step - time_adjust
                start_line = - time_adjust
                if args.alg == 'lr-l2':
                    if word == 'noun1':
                        time_select = np.logical_and(time_win >= -0.5, time_win <= 4.0)
                    elif word == 'verb':
                        time_select = np.logical_and(time_win >= -1.5, time_win <= 3.0)
                    elif word == 'noun2':
                        time_select = np.logical_and(time_win >= -3.0, time_win <= 1.5)
                        # print(len(time_select))
                else:
                    if word == 'verb':
                        max_line -= 1.0
                        start_line -= 1.0
                    else:
                        max_line -= 2.5
                        start_line -= 2.5
            mean_acc = mean_acc[time_select, :]
            mean_acc = mean_acc[:, time_select]
            time_win = time_win[time_select]
            print(mean_acc.shape)
            print(np.max(mean_acc))
            num_time = len(time_win)

            ax = combo_grid[i_combo]
            print(i_combo)
            im = ax.imshow(mean_acc, interpolation='nearest', aspect='auto', vmin=0.25, vmax=0.75)

            ax.set_title('{word}'.format(
                word=PLOT_TITLE_WORD[word]), fontsize=axistitlesize)

            ax.set_xticks(np.arange(0.0, float(num_time), float(time_step)) - time_adjust)
            ax.set_yticks(np.arange(0, num_time, time_step) - time_adjust)
            if i_combo == len(word_list)*2 - 1:
                # ax.set_xticks(np.arange(0.0, float(num_time), float(time_step)))
                # ax.set_yticks(np.arange(0, num_time, time_step))
                ax.set_xlim([0.0, float(num_time)])
                ax.set_ylim([float(num_time), 0.0])

            min_time = 0.0
            max_time = 0.5 * len(time_win) / time_step
            label_time = np.arange(min_time, max_time, 0.5)
            ax.set_xticklabels(label_time)

            ax.set_yticklabels(label_time)
            ax.tick_params(labelsize=ticklabelsize)
            for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
                ax.axvline(x=v, color='w')
                if i_v == 0:
                    buff_space = 0.125
                else:
                    buff_space = 0.025
                if i_v < len(text_to_write):
                    ax.text(v + buff_space * 2*time_step, 0.75*time_step, text_to_write[i_v], color='w', fontsize=10)
            ax.text(-0.12, 1.02, string.ascii_uppercase[i_combo], transform=ax.transAxes,
                                    size=axislettersize, weight='bold')
            i_combo += 1
        


    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('TGM Averaged Over Subjects',
        fontsize=suptitlesize)
    combo_fig.text(0.04, 0.275, 'Train Time Relative to Sentence Onset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    combo_fig.text(0.5, 0.04, 'Test Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.text(0.488, 0.8575, 'Active', ha='center', fontsize=axistitlesize+2)
    combo_fig.text(0.488, 0.475, 'Passive', ha='center', fontsize=axistitlesize+2)
    combo_fig.subplots_adjust(top=0.85)
    combo_fig.savefig('/home/nrafidi/thesis_figs/{exp}_avg-tgm_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                    exp=args.experiment, sen_type='both', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    num_instances=args.num_instances
                ), bbox_inches='tight')

    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_avg-tgm_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type='both', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    plt.show()

# Boneyard

# fig, ax = plt.subplots()
            # h = ax.imshow(np.squeeze(intersection), interpolation='nearest', aspect='auto', vmin=0, vmax=acc_all.shape[0])
            #
            # ax.set_ylabel('Train Time (s)')
            # ax.set_xlabel('Test Time (s)')
            # ax.set_title(
            #     'Intersection TGM\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
            #                                                               word=word,
            #                                                               experiment=args.experiment))
            # ax.set_xticks(range(0, len(time_win), time_step))
            # label_time = time_win
            # label_time = label_time[::time_step]
            # label_time[np.abs(label_time) < 1e-15] = 0.0
            # ax.set_xticklabels(label_time)
            # ax.set_yticks(range(0, len(time_win), time_step))
            # ax.set_yticklabels(label_time)
            # ax.set_xlim(left=time_step)
            # ax.set_ylim(top=time_step)
            #
            # for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
            #     ax.axvline(x=v, color='k')
            #     if i_v < len(text_to_write):
            #         plt.text(v + 0.05 * 2*time_step, time_step, text_to_write[i_v], color='k')
            # plt.colorbar(h)
            #
            # fig.tight_layout()
            # plt.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_intersection_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            #         win_len=args.win_len,
            #         overlap=args.overlap,
            #         num_instances=args.num_instances
            #     ), bbox_inches='tight')

# fig, ax = plt.subplots()
            # h = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=0.25, vmax=0.5)
            # ax.set_ylabel('Train Time (s)')
            # ax.set_xlabel('Test Time (s)')
            # ax.set_title('Average TGM Decoding {word} from {sen_type}\n{avgTime}, {avgTest}\nNumber of Instances: {ni}'.format(
            #     sen_type=PLOT_TITLE_SEN[sen_type],
            #     word=PLOT_TITLE_WORD[word],
            #     avgTime=avg_time_str,
            #     avgTest=avg_test_str,
            #     ni=args.num_instances))
            # ax.set_xticks(range(0, len(time_win), time_step))
            # label_time = time_win
            # label_time = label_time[::time_step]
            # label_time[np.abs(label_time) < 1e-15] = 0.0
            # ax.set_xticklabels(label_time)
            # ax.set_yticks(range(0, len(time_win), time_step))
            # ax.set_yticklabels(label_time)
            # time_adjust = args.win_len
            #
            # for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
            #     ax.axvline(x=v, color='w')
            #     if i_v < len(text_to_write):
            #         plt.text(v + 0.05 * 2 * time_step, 1.5 * time_step, text_to_write[i_v], color='w')
            # plt.colorbar(h)
            # ax.set_xlim(left=time_step)
            # ax.set_ylim(top=time_step)
            # fig.tight_layout()
            # plt.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_avg-tgm-title_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            #         win_len=args.win_len,
            #         overlap=args.overlap,
            #         num_instances=args.num_instances
            #     ), bbox_inches='tight')
            #
            #
            #
            #
            # time_adjust = args.win_len*0.002
            # fig, ax = plt.subplots()
            #
            # ax.plot(np.diag(mean_acc), label='Accuracy')
            # ax.plot(frac_sub, label='Fraction of Subjects > Chance')
            #
            #
            # ax.set_xticks(range(0, len(time_win), time_step))
            # label_time = time_win
            # label_time = label_time[::time_step]
            # label_time[np.abs(label_time) < 1e-15] = 0.0
            # ax.set_xticklabels(label_time)
            # for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
            #     ax.axvline(x=v, color='k')
            #     if i_v < len(text_to_write):
            #         plt.text(v + 0.05, 0.8, text_to_write[i_v])
            # ax.set_ylabel('Accuracy/Fraction > Chance')
            # ax.set_xlabel('Time')
            # ax.set_ylim([0.0, 1.0])
            # ax.set_xlim(left=time_step)
            # ax.legend(loc=4)
            # ax.set_title('Mean Acc over subjects and Frac > Chance\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
            #                                                                              word=word,
            #                                                                              experiment=args.experiment))
            #
            # fig.tight_layout()
            # plt.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_diag_acc_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            #         win_len=args.win_len,
            #         overlap=args.overlap,
            #         num_instances=args.num_instances
            #     ), bbox_inches='tight')


# if sen_type == 'active':
#     text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
#     max_line = 2.0 - time_adjust + 0.01
#     if word == 'noun1':
#         start_line = -0.5 - time_adjust
#     else:
#         start_line = -1.0 - time_adjust
#
# else:
#     text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
#     max_line = 3.0 - time_adjust + 0.01
#     if word == 'noun1':
#         start_line = -0.5 - time_adjust
#     else:
#         start_line = -1.5 - time_adjust
