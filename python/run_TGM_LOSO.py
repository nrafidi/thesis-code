import argparse
import load_data_ordered as load_data
import models
import numpy as np
import os.path
import random
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_{mode}'

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


# Runs the TGM experiment
def run_tgm_exp(experiment,
                subject,
                sen_type,
                word,
                win_len,
                overlap,
                isPerm = False,
                alg='lr-l1',
                adj=None,
                doTimeAvg=False,
                doTestAvg=True,
                num_instances=1,
                reps_to_use=10,
                proc=load_data.DEFAULT_PROC,
                random_state_perm=1,
                force=False,
                mode='acc'):
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
                             mode=mode)


    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return

    data, labels, time, final_inds = load_data.load_sentence_data(subject=subject,
                                                                  word='noun1',
                                                                  sen_type=sen_type,
                                                                  experiment=experiment,
                                                                  proc=proc,
                                                                  num_instances=num_instances,
                                                                  reps_to_use=reps_to_use,
                                                                  noMag=False,
                                                                  sorted_inds=None)

    print(labels)

    tmin = time.min()
    tmax = time.max()

    total_win = int((tmax - tmin) * 500)

    if win_len < 0:
        win_len = total_win - overlap

    win_starts = range(0, total_win - win_len, overlap)

    sen_set = np.unique(labels, axis=0).tolist()
    print(sen_set)
    num_labels = labels.shape[0]
    sen_ints = np.empty((num_labels,))
    for i_l in range(num_labels):
        for j_l, l in enumerate(sen_set):
            print(l)
            print(labels[i_l, :])
            if np.all(l == labels[i_l, :]):
                sen_ints[i_l] = j_l
                break

    print(sen_ints)
    labels = labels[:, WORD_COLS[experiment][word]]

    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)

    if mode == 'acc':
        l_ints, cv_membership, tgm_acc = models.lr_tgm_loso(data,
                                                            labels,
                                                            win_starts,
                                                            win_len,
                                                            sen_ints,
                                                            penalty=alg[3:],
                                                            adj=adj,
                                                            doTimeAvg=doTimeAvg,
                                                            doTestAvg=doTestAvg)
        np.savez_compressed(fname,
                            l_ints=l_ints,
                            cv_membership=cv_membership,
                            tgm_acc=tgm_acc,
                            win_starts=win_starts,
                            time=time,
                            proc=proc)
    else:
        l_ints, coef, Cs = models.lr_tgm_coef(data,
                                              labels,
                                              win_starts,
                                              win_len,
                                              penalty=alg[3:],
                                              adj=adj,
                                              doTimeAvg=doTimeAvg)
        np.savez_compressed(fname,
                            l_ints=l_ints,
                            coef=coef,
                            Cs=Cs,
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
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--force', default='False', choices=['True', 'False'])
    parser.add_argument('--mode', choices=['acc', 'coef'])

    args = parser.parse_args()
    print(args)
    # Check that parameter setting is valid
    total_valid = True
    is_valid = args.reps_to_use <= load_data.NUM_REPS[args.experiment]
    total_valid = total_valid and is_valid
    if not is_valid:
        print('num reps  wrong')
    is_valid = args.subject in load_data.VALID_SUBS[args.experiment]
    total_valid = total_valid and is_valid
    if not is_valid:
        print('subject wrong')
    if args.num_instances != 2:
        is_valid = (args.reps_to_use % args.num_instances) == 0
        total_valid = total_valid and is_valid
        if not is_valid:
            print('instances wrong')

    if total_valid:
        run_tgm_exp(experiment=args.experiment,
                    subject=args.subject,
                    sen_type=args.sen_type,
                    word=args.word,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    isPerm=str_to_bool(args.isPerm),
                    alg=args.alg,
                    adj=args.adj,
                    doTimeAvg=str_to_bool(args.doTimeAvg),
                    doTestAvg=str_to_bool(args.doTestAvg),
                    num_instances=args.num_instances,
                    reps_to_use=args.reps_to_use,
                    proc=args.proc,
                    random_state_perm=args.perm_random_state,
                    force=str_to_bool(args.force),
                    mode=args.mode)
    else:
        print('Experiment parameters not valid. Skipping job.')
