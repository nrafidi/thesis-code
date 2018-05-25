import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO_EOS
from mpl_toolkits.axes_grid1 import AxesGrid
import string

PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice',
                   'propid': 'Proposition ID',
                   'senlen': 'Sentence Length'}


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'
CHANCE = {'krns2':{'pooled': {'noun1': 0.125,
                             'verb': 0.25,
                             'voice': 0.5,
                              'agent': 0.25,
                              'patient': 0.25,
                              'propid': 1.0/16.0},
                  'active': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                             'agent': 0.25,
                             'patient': 0.25,
                             'propid': 1.0 / 8.0
                             },
                  'passive': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                              'agent': 0.25,
                              'patient': 0.25,
                              'propid': 1.0 / 8.0
                              }
                    },
          'PassAct3': {'pooled': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                                  'agent': 0.25,
                                  'patient': 0.25,
                              'propid': 1.0/8.0,
                                  'senlen': 0.5},
                  'active': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5},
                  'passive': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5}
                    }}

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
    top_dir = run_TGM_LOSO_EOS.TOP_DIR.format(exp=exp)

    chance=CHANCE[exp][sen_type][word]

    acc_by_sub = []
    acc_intersect = []
    time_by_sub = []
    win_starts_by_sub = []
    eos_max_by_sub = []
    for sub in load_data.VALID_SUBS[exp]:
        save_dir = run_TGM_LOSO_EOS.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result_fname = run_TGM_LOSO_EOS.SAVE_FILE.format(dir=save_dir,
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
                                                         rsP=1,
                                                         mode='acc') + '.npz'
        if not os.path.isfile(result_fname):
            print(result_fname)
            continue
        result = np.load(result_fname)
        time = np.squeeze(result['time'])
        win_starts = result['win_starts']

        fold_acc = result['tgm_acc']
        eos_max_fold = []
        for i_fold in range(fold_acc.shape[0]):
            diag_acc = np.diag(np.squeeze(fold_acc[i_fold, :, :]))
            argo = np.argmax(diag_acc)
            eos_max_fold.append(argo)
        eos_max_fold = np.array(eos_max_fold)
        eos_max_by_sub.append(eos_max_fold[None, :])
        acc = np.mean(fold_acc, axis=0)

        time_by_sub.append(time[None, ...])
        win_starts_by_sub.append(win_starts[None, ...])
        acc_thresh = acc > chance
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
    # parser.add_argument('--sen_type', choices=run_TGM_LOSO_EOS.VALID_SEN_TYPE)
    # parser.add_argument('--word', choices = ['noun1', 'verb', 'voice', 'agent', 'patient'])
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='T')
    args = parser.parse_args()

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    if args.experiment == 'krns2':
        word_list = ['noun1', 'agent', 'patient', 'verb', 'voice', 'propid']
    else:
        word_list = word_list = ['voice', 'senlen', 'verb', 'agent', 'patient', 'propid']
    num_plots = len(word_list)/2
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(num_plots*6, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, num_plots),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5)
    sen_type = 'pooled'
    for i_word, word in enumerate(word_list):
        chance = CHANCE[args.experiment][sen_type][word]
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
        mean_acc -= chance
        if np.all(mean_acc <= 0.0):
            mean_acc = np.zeros(mean_acc.shape)
        else:
            mean_acc /= np.max(mean_acc)

        ax = combo_grid[i_word]
        im = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0)

        ax.set_title('{word}'.format(
            word=PLOT_TITLE_WORD[word]), fontsize=axistitlesize)
        time_win = time[win_starts]
        num_time = len(time_win)
        ax.set_xticks(np.arange(0, num_time, time_step) - time_adjust)
        min_time = 0.0
        max_time = 0.5 * len(time_win) / time_step
        label_time = np.arange(min_time, max_time, 0.5)

        ax.set_xticklabels(label_time)
        ax.set_yticks(np.arange(0, num_time, time_step) - time_adjust)
        ax.set_yticklabels(label_time)
        ax.tick_params(labelsize=ticklabelsize)
        ax.axvline(x=0.625*time_step, color='w')
        ax.text(-0.15, 1.05, string.ascii_uppercase[i_word], transform=ax.transAxes,
                size=axislettersize, weight='bold')


        time_adjust = args.win_len*0.002


    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('TGM Averaged Over Subjects',
                       fontsize=suptitlesize)
    combo_fig.text(0.04, 0.275, 'Train Time Relative to Last Word Onset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    combo_fig.text(0.5, 0.04, 'Test Time Relative to Last Word Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=args.experiment, sen_type='pooled', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=args.win_len,
                overlap=args.overlap,
                num_instances=args.num_instances
            ), bbox_inches='tight')
    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type='pooled', word='all', alg=args.alg, avgTime=args.avgTime,
            avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')
    plt.show()

# boneyard

    # fig, ax = plt.subplots()
    # h = ax.imshow(np.squeeze(intersection), interpolation='nearest', aspect='auto', vmin=0, vmax=acc_all.shape[0])
    # ax.set_ylabel('Test Time')
    # ax.set_xlabel('Train Time')
    # ax.set_title(
    #     'Intersection TGM\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
    #                                                               word=word,
    #                                                               experiment=args.experiment))
    # ax.set_xticks(range(0, len(time[win_starts]), time_step))
    # label_time = time[win_starts]
    # label_time = label_time[::time_step]
    # label_time[np.abs(label_time) < 1e-15] = 0.0
    # ax.set_xticklabels(label_time)
    # ax.set_yticks(range(0, len(time[win_starts]), time_step))
    # ax.set_yticklabels(label_time)
    #
    # plt.colorbar(h)
    #
    # fig.tight_layout()
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_eos_intersection_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
    #         win_len=args.win_len,
    #         overlap=args.overlap,
    #         num_instances=args.num_instances
    #     ), bbox_inches='tight')

    # fig, ax = plt.subplots()
    # ax.plot(time[win_starts], np.diag(mean_acc), label='Accuracy')
    # ax.plot(time[win_starts], frac_sub, label='Fraction of Subjects > Chance')
    #
    # ax.set_ylabel('Accuracy/Fraction > Chance')
    # ax.set_xlabel('Time')
    # ax.set_ylim([0.0, 1.0])
    # ax.legend(loc=4)
    # ax.set_title('Mean Acc over subjects and Frac > Chance\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
    #                                                                                                word=word,
    #                                                                                                experiment=args.experiment))
    #
    # fig.tight_layout()
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
    #         win_len=args.win_len,
    #         overlap=args.overlap,
    #         num_instances=args.num_instances
    #     ), bbox_inches='tight')

    # for sub in range(acc_all.shape[0]):
    #     fig, ax = plt.subplots()
    #     h = ax.imshow(np.squeeze(acc_all[sub, ...]), interpolation='nearest', aspect='auto', vmin=0, vmax=1.0)
    #     ax.set_ylabel('Test Time')
    #     ax.set_xlabel('Train Time')
    #     ax.set_title('Subject {sub} TGM\n{sen_type} {word} {experiment}'.format(sub=load_data.VALID_SUBS[args.experiment][sub],
    #                                                                             sen_type=sen_type,
    #                                                                             word=word,
    #                                                                             experiment=args.experiment))
    #     ax.set_xticks(range(0, len(time[win_starts]), time_step))
    #     label_time = time[win_starts]
    #     label_time = label_time[::time_step]
    #     label_time[np.abs(label_time) < 1e-15] = 0.0
    #     ax.set_xticklabels(label_time)
    #     ax.set_yticks(range(0, len(time[win_starts]), time_step))
    #     ax.set_yticklabels(label_time)
    #     time_adjust = args.win_len
    #
    #     plt.colorbar(h)

# plt.colorbar(h)
        #
        # fig.tight_layout()
        # plt.savefig(
        #     '/home/nrafidi/thesis_figs/{exp}_eos_avgtgm_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
        #         exp=args.experiment, sen_type=sen_type, word=word, avgTime=args.avgTime, avgTest=args.avgTest,
        #         win_len=args.win_len,
        #         overlap=args.overlap,
        #         num_instances=args.num_instances
        #     ), bbox_inches='tight')

        # fig, ax = plt.subplots()
        # ax.plot(np.diag(intersection))
        # ax.set_ylabel('Number of Subjects with > chance acc')
        # ax.set_xlabel('Time')
        # ax.set_title('Intersection of acc > chance over subjects\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
        #                                                                              word=word,
        #                                                                              experiment=args.experiment))

        # fig, ax = plt.subplots()
        # h = ax.imshow(mean_acc, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        # ax.set_ylabel('Test Time')
        # ax.set_xlabel('Train Time')
        # ax.set_title('Mean Acc TGM over subjects\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
        #                                                                                  word=word,
        #                                                                                  experiment=args.experiment))
        # plt.colorbar(h)