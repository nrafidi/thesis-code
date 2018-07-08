import argparse
import numpy as np
import string
import os
from rank_from_pred import rank_from_pred


TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
MULTI_SAVE_FILE = '{dir}TGM-LOSO{tsss}_multisub-subcv_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}{rep_str}_{rank_str}{mode}'

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
    parser.add_argument('--tsss', action='store_true')
    parser.add_argument('--num_reps', type=int, default=10)
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

    ticklabelsize = 12
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    top_dir = TOP_DIR.format(exp=args.experiment)

    if args.tsss:
        tsss_str = '_tsss'
    else:
        tsss_str = ''

    if args.num_reps < 10:
        rep_str = '_nr{nr}'.format(nr=args.num_reps)
    else:
        rep_str = ''

    word_list = ['noun1', 'verb', 'noun2']
    num_plots = len(word_list)
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    i_combo = 0
    for i_sen, sen_type in enumerate(['active', 'passive']):
        for i_word, word in enumerate(word_list):

            multi_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                                sen_type=sen_type,
                                                word=word,
                                                win_len=args.win_len,
                                                tsss=tsss_str,
                                                ov=args.overlap,
                                                perm='F',
                                                alg=args.alg,
                                                adj=args.adj,
                                                avgTm=args.avgTime,
                                                avgTst=args.avgTest,
                                                inst=args.num_instances,
                                                rsP=1,
                                                rep_str=rep_str,
                                                rank_str='',
                                                mode='acc')
            rank_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                                sen_type=sen_type,
                                                word=word,
                                                win_len=args.win_len,
                                                tsss=tsss_str,
                                                ov=args.overlap,
                                                perm='F',
                                                alg=args.alg,
                                                adj=args.adj,
                                                avgTm=args.avgTime,
                                                avgTst=args.avgTest,
                                                inst=args.num_instances,
                                                rsP=1,
                                               rep_str=rep_str,
                                                rank_str='rank',
                                                mode='acc')

            result = np.load(multi_file + '.npz')
            # rerun = args.long and sen_type == 'passive'
            if os.path.isfile(rank_file + '.npz'): # and not rerun:
                rank_result = np.load(rank_file + '.npz')
                acc_all = rank_result['tgm_rank']
            else:
                tgm_pred = result['tgm_pred']
                print(tgm_pred.shape)
                print(word)
                assert len(tgm_pred.shape) == 4
                num_sub = tgm_pred.shape[0]
                l_ints = result['l_ints']
                print(l_ints.shape)
                cv_membership_all = result['cv_membership']
                print(len(cv_membership_all))
                acc_all = []
                for i_sub in range(num_sub):
                    cv_membership = cv_membership_all[i_sub]
                    print(len(cv_membership))
                    fold_labels = []
                    for i in range(len(cv_membership)):
                        fold_labels.append(np.mean(l_ints[cv_membership[i]]))

                    tgm_rank = rank_from_pred(tgm_pred[i_sub, ...], fold_labels)
                    acc_all.append(tgm_rank[None, ...])
                acc_all = np.concatenate(acc_all, axis=0)
                print(acc_all.shape)
                np.savez_compressed(rank_file, tgm_rank=acc_all)