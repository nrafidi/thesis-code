import argparse
from syntax_vs_semantics import load_data
import models
import numpy as np
import os.path
import random
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_alg_comp/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-alg-comp_{sub}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}'
NUM_SENTENCES = 16
NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']}
VALID_ALGS = ['lr-None', 'lr-l2', 'lr-l1', 'svm-l2', 'svm-l1', 'gnb', 'gnb-None']
VALID_WORDS = ['verb', 'voice']


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
    else:
        return str_thing

def tgm_from_preds_GNB(preds, l_ints, cv_membership):
    num_folds = preds.shape[0]
    num_win = preds.shape[1]
    tgm_corr = np.zeros((num_win, num_win))
    tgm_total = np.zeros((num_win, num_win))
    for fold in range(num_folds):
        labels = l_ints[cv_membership[fold]]
        # print(labels)
        for i_win in range(num_win):
            for j_win in range(num_win):
                yhat = np.argmax(preds[fold, i_win, j_win], axis=1)
                # print(yhat.shape)
                tgm_corr[i_win, j_win] += np.sum(yhat == labels)
                tgm_total[i_win, j_win] += preds[fold, i_win, j_win].shape[0]
    tgm = np.divide(tgm_corr, tgm_total)
    return tgm


# Runs the TGM experiment
def run_tgm_exp(experiment,
                subject,
                word,
                win_len,
                overlap,
                isPerm = False,
                alg='lr-l1',
                adj=None,
                doTimeAvg=False,
                doTestAvg=True,
                num_instances=1,
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
                             word=word,
                             win_len=win_len,
                             ov=overlap,
                             perm=bool_to_str(isPerm),
                             alg=alg,
                             adj=adj,
                             avgTm=bool_to_str(doTimeAvg),
                             avgTst=bool_to_str(doTestAvg),
                             inst=num_instances,
                             rsP=random_state_perm)

    print(force)
    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return


    if word == 'voice':
        data, _, sen_ints, time, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                     align_to='last',
                                                                     voice=['active', 'passive'],
                                                                     experiment=experiment,
                                                                     proc=proc,
                                                                     num_instances=num_instances,
                                                                     reps_filter=None,
                                                                     sensor_type=None,
                                                                     is_region_sorted=False,
                                                                     tmin=0.5,
                                                                     tmax=1.0)
    else:
        data_act, _, sen_ints_act, time, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                     align_to='verb',
                                                                     voice=['active'],
                                                                     experiment=experiment,
                                                                     proc=proc,
                                                                     num_instances=num_instances,
                                                                     reps_filter=None,
                                                                     sensor_type=None,
                                                                     is_region_sorted=False,
                                                                     tmin=0.0,
                                                                     tmax=0.5)
        data_pass, _, sen_ints_pass, _, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                             align_to='verb',
                                                                             voice=['passive'],
                                                                             experiment=experiment,
                                                                             proc=proc,
                                                                             num_instances=num_instances,
                                                                             reps_filter=None,
                                                                             sensor_type=None,
                                                                             is_region_sorted=False,
                                                                             tmin=-0.5,
                                                                             tmax=0.0)

        data = np.concatenate([data_act, data_pass], axis=0)
        sen_ints = np.concatenate([sen_ints_act, sen_ints_pass], axis=0)

    stimuli_voice = list(load_data.read_stimuli(experiment))
    labels = []
    for i_sen_int, sen_int in enumerate(sen_ints):
        curr_voice = stimuli_voice[sen_int]['voice']
        labels.append(curr_voice)

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

    if 'lr' in alg:
        l_ints, cv_membership, tgm_acc, tgm_pred = models.lr_tgm_loso(data,
                                                                      labels,
                                                                      win_starts,
                                                                      win_len,
                                                                      sen_ints,
                                                                      penalty=str_to_none(alg[3:]),
                                                                      adj=adj,
                                                                      doTimeAvg=doTimeAvg,
                                                                      doTestAvg=doTestAvg)
    elif 'svm' in alg:
        l_ints, cv_membership, tgm_acc, tgm_pred = models.svc_tgm_loso(data,
                 labels,
                 win_starts,
                 win_len,
                 sen_ints,
                 sub_rs=1,
                 penalty=alg[4:],
                 adj=adj,
                 doTimeAvg=doTimeAvg,
                 doTestAvg=doTestAvg,
                 ddof=1,
                 C=None)
    else:
        if adj == 'zscore':
            doZscore=True
        else:
            doZscore=False
        if 'None' in alg:
            doFeatSelect=False
        else:
            doFeatSelect=True
        tgm_pred, l_ints, cv_membership, feature_masks, num_feat_selected = models.nb_tgm_loso(data,
                labels,
                sen_ints,
                1,
                win_starts,
                win_len,
                feature_select=doFeatSelect,
                doZscore=doZscore,
                doAvg=doTimeAvg,
                ddof=1)
        tgm_acc = tgm_from_preds_GNB(tgm_pred, l_ints, cv_membership)

    print('Max Acc: %.2f' % np.max(np.mean(tgm_acc, axis=0)))
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
    parser.add_argument('--word', choices=VALID_WORDS)
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--isPerm', default='False', choices=['True', 'False'])
    parser.add_argument('--alg', default='lr-l1', choices=VALID_ALGS)
    parser.add_argument('--adj', default=None)
    parser.add_argument('--doTimeAvg', default='False', choices=['True', 'False'])
    parser.add_argument('--doTestAvg', default='False', choices=['True', 'False'])
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--force', default='False', choices=['True', 'False'])

    args = parser.parse_args()
    print(args)
    # Check that parameter setting is valid
    total_valid = True
    is_valid = args.subject in VALID_SUBS[args.experiment]
    total_valid = total_valid and is_valid
    if not is_valid:
        print('subject wrong')

    if total_valid:
        run_tgm_exp(experiment=args.experiment,
                    subject=args.subject,
                    word=args.word,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    isPerm=str_to_bool(args.isPerm),
                    alg=args.alg,
                    adj=args.adj,
                    doTimeAvg=str_to_bool(args.doTimeAvg),
                    doTestAvg=str_to_bool(args.doTestAvg),
                    num_instances=args.num_instances,
                    proc=args.proc,
                    random_state_perm=args.perm_random_state,
                    force=str_to_bool(args.force))
    else:
        print('Experiment parameters not valid. Skipping job.')
