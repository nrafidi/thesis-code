import argparse
from syntax_vs_semantics import load_data
import models
import numpy as np
import os.path
import random
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_{fold}'
NUM_SENTENCES = 16
NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']}
VALID_ALGS = ['lr-l2', 'lr-l1']
VALID_SEN_TYPE = ['active', 'passive']

WORD_COLS = {'krns2': {'art1': 0,
                       'noun1': 1,
                       'verb': 2,
                       'art2': 3,
                       'noun2': 4},
             'PassAct2': {'noun1': 0,
                          'verb': 1,
                          'noun2': 2},
             'PassAct3': {'noun1': 0,
                          'verb': 1,
                          'noun2': 2}
             }


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'

def str_to_bool(str_bool):
    if str_bool == 'False':
        return False
    else:
        return True

def str_to_none(str_thing):
    if str_thing =='None':
        return None


# Runs the TGM experiment
def run_tgm_exp(experiment,
                subject,
                sen_type,
                word,
                win_len,
                overlap,
                fold,
                isPerm = False,
                alg='lr-l1',
                adj=None,
                doTimeAvg=False,
                doTestAvg=True,
                num_instances=1,
                reps_to_use=10,
                proc=load_data.DEFAULT_PROC,
                random_state_perm=1,
                force=False):
    warnings.filterwarnings(action='ignore')
    # Save Directory
    top_dir = TOP_DIR.format(exp=experiment)
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)
    save_dir = SAVE_DIR.format(top_dir=top_dir, sub=subject)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname = SAVE_FILE.format(dir=save_dir,
                             sub=subject,
                             sen_type=sen_type,
                             word=word,
                             win_len=win_len,
                             ov=overlap,
                             perm=bool_to_str(isPerm),
                             alg=alg,
                             adj=adj,
                             avgTm=bool_to_str(doTimeAvg),
                             avgTst=bool_to_str(doTestAvg),
                             inst=num_instances,
                             rep=reps_to_use,
                             rsP=random_state_perm,
                             fold=fold)

    print(force)
    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return

    data, labels, sen_ints, time, sensor_regions = load_data.load_sentence_data_v2(subject=subject,
                                                                                   align_to=word,
                                                                                   voice=sen_type,
                                                                                   experiment=experiment,
                                                                                   proc=proc,
                                                                                   num_instances=num_instances,
                                                                                   reps_filter=reps_to_use,
                                                                                   sensor_type=None,
                                                                                   is_region_sorted=False,
                                                                                   tmin=None,
                                                                                   tmax=None)
    print(data.shape)
    print(labels)

    tmin = time.min()
    tmax = time.max()

    total_win = int((tmax - tmin) * 500)

    if win_len < 0:
        win_len = total_win - overlap

    win_starts = range(0, total_win - win_len, overlap)


    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)


    l_ints, cv_membership, tgm_acc, tgm_pred = models.lr_tgm_loso_fold(data,
                                                                        labels,
                                                                        win_starts,
                                                                        win_len,
                                                                        sen_ints,
                                                                       fold,
                                                                        penalty=alg[3:],
                                                                        adj=adj,
                                                                        doTimeAvg=doTimeAvg,
                                                                        doTestAvg=doTestAvg)
    np.savez_compressed(fname,
                        l_ints=l_ints,
                        cv_membership=cv_membership,
                        tgm_acc=tgm_acc,
                        tgm_pred=tgm_pred,
                        win_starts=win_starts,
                        time=time,
                        proc=proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--sen_type', choices=VALID_SEN_TYPE)
    parser.add_argument('--word', choices=['noun1', 'noun2', 'verb'])
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--isPerm', default='False', choices=['True', 'False'])
    parser.add_argument('--alg', default='lr-l1', choices=VALID_ALGS)
    parser.add_argument('--adj', default=None)
    parser.add_argument('--doTimeAvg', default='False', choices=['True', 'False'])
    parser.add_argument('--doTestAvg', default='False', choices=['True', 'False'])
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use')
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--force') #, default='False', choices=['True', 'False'])
    parser.add_argument('--fold', type=int)

    args = parser.parse_args()
    print(args)
    # Check that parameter setting is valid
    total_valid = True
    # is_valid = args.reps_to_use <= NUM_REPS[args.experiment]
    # total_valid = total_valid and is_valid
    # if not is_valid:
    #     print('num reps  wrong')
    is_valid = args.subject in VALID_SUBS[args.experiment]
    total_valid = total_valid and is_valid
    if not is_valid:
        print('subject wrong')

    if total_valid:
        run_tgm_exp(experiment=args.experiment,
                    subject=args.subject,
                    sen_type=args.sen_type,
                    word=args.word,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    fold=args.fold,
                    isPerm=str_to_bool(args.isPerm),
                    alg=args.alg,
                    adj=args.adj,
                    doTimeAvg=str_to_bool(args.doTimeAvg),
                    doTestAvg=str_to_bool(args.doTestAvg),
                    num_instances=args.num_instances,
                    reps_to_use=str_to_none(args.reps_to_use),
                    proc=args.proc,
                    random_state_perm=args.perm_random_state,
                    force=str_to_bool(args.force))
    else:
        print('Experiment parameters not valid. Skipping job.')
