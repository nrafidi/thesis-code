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

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'
MULTI_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'


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
                             'voice': 0.5,
                                  'agent': 0.25,
                                  'patient': 0.25,
                              'propid': 1.0/8.0,},
                  'passive': {'noun1': 0.25,
                             'verb': 0.25,
                             'voice': 0.5,
                                  'agent': 0.25,
                                  'patient': 0.25,
                              'propid': 1.0/8.0,}
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
    for sub in run_TGM_LOSO_EOS.VALID_SUBS[exp]:
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
    parser.add_argument('--sen_type', choices=run_TGM_LOSO_EOS.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['noun1', 'verb', 'voice', 'agent', 'patient'])
    parser.add_argument('--win_len', type=int, default=25)
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
        vmaxes = {'noun1': 0.2,
                  'agent': 0.4,
                  'patient': 0.4,
                  'verb': 0.4,
                  'voice': 0.8,
                  'propid': 0.2}
    else:
        vmaxes = {'noun1': 0.4,
                  'agent': 0.35,
                  'patient': 0.35,
                  'verb': 0.35,
                  'voice': 0.75,
                  'senlen': 0.75,
                  'propid': 0.25}

    sen_type = args.sen_type
    word = args.word


    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(16, 8))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(1, 2),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.2)
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

    mean_acc = np.mean(acc_all, axis=0)

    ax = combo_grid[0]
    im = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=chance, vmax=vmaxes[word])

    ax.set_title('Average over Single Subjects', fontsize=axistitlesize)
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
    ax.text(-0.15, 1.05, string.ascii_uppercase[0], transform=ax.transAxes,
            size=axislettersize, weight='bold')

    top_dir = TOP_DIR.format(exp=args.experiment)
    multi_file = MULTI_SAVE_FILE.format(dir=top_dir,
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

    result = np.load(multi_file)
    multi_fold_acc = result['tgm_acc']

    multi_mean_acc = np.mean(multi_fold_acc, axis=0)

    ax = combo_grid[1]
    im = ax.imshow(np.squeeze(multi_mean_acc), interpolation='nearest', aspect='auto', vmin=chance, vmax=vmaxes[word])

    ax.set_title('Multi-Subject', fontsize=axistitlesize)
    ax.set_xticks(np.arange(0, num_time, time_step) - time_adjust)
    ax.set_xticklabels(label_time)
    ax.set_yticks(np.arange(0, num_time, time_step) - time_adjust)
    ax.set_yticklabels(label_time)
    ax.tick_params(labelsize=ticklabelsize)
    ax.axvline(x=0.625 * time_step, color='w')
    ax.text(-0.15, 1.05, string.ascii_uppercase[1], transform=ax.transAxes,
            size=axislettersize, weight='bold')

    cbar = combo_grid.cbar_axes[0].colorbar(im)
    time_adjust = args.win_len*0.002

    combo_fig.suptitle('TGM decoding {word} from {sen_type}'.format(word=word, sen_type=sen_type),
                       fontsize=suptitlesize)
    combo_fig.text(0.04, 0.275, 'Train Time Relative to Last Word Onset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    combo_fig.text(0.5, 0.04, 'Test Time Relative to Last Word Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm-multi-comp_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=args.win_len,
                overlap=args.overlap,
                num_instances=args.num_instances
            ), bbox_inches='tight')
    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_avg-tgm-multi-comp_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime,
            avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')
    plt.show()