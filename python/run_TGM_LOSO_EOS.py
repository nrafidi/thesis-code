import argparse
from syntax_vs_semantics import load_data
import models
import numpy as np
import os.path
import random
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'
NUM_SENTENCES = 16
NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']}
VALID_ALGS = ['lr-l2', 'lr-l1']
VALID_SEN_TYPE = ['active', 'passive', 'pooled']
VALID_WORDS = ['noun1', 'noun2', 'verb', 'voice', 'agent', 'patient', 'propid', 'senlen']

TMAX={'krns2': 1.5,
      'PassAct2': 1.5,
      'PassAct3': 2.0}

WORD_COLS = {'active': {'noun1': 1,
                        'verb': 2,
                        'noun2': 4,
                        'agent': 1,
                        'patient': 4},
             'passive': {'noun1': 1,
                         'verb': 3,
                         'noun2': 6,
                         'agent': 6,
                         'patient': 1}}


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
                isPerm = False,
                alg='lr-l1',
                adj=None,
                doTimeAvg=False,
                doTestAvg=True,
                num_instances=1,
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
                             rsP=random_state_perm,
                             mode=mode)

    print(force)
    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return

    if sen_type == 'pooled':
        voice = ['active', 'passive']
    else:
        voice = [sen_type]

    data, _, sen_ints, time, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                 align_to='last',
                                                                 voice=voice,
                                                                 experiment=experiment,
                                                                 proc=proc,
                                                                 num_instances=num_instances,
                                                                 reps_filter=None,
                                                                 sensor_type=None,
                                                                 is_region_sorted=False,
                                                                 tmin=0.0,
                                                                 tmax=TMAX[experiment])

    stimuli_voice = list(load_data.read_stimuli(experiment))
    # print(stimuli_voice)
    if word == 'propid':
        all_words = [stimuli_voice[sen_int]['stimulus'].split() for sen_int in sen_ints]
        all_voices = [stimuli_voice[sen_int]['voice'] for sen_int in sen_ints]
        content_words = []
        valid_inds = []
        for i_word_list, word_list in enumerate(all_words):
            curr_voice = all_voices[i_word_list]
            if experiment == 'PassAct3':
                if len(word_list) > 5:
                    valid_inds.append(i_word_list)
                    content_words.append([word_list[WORD_COLS[curr_voice]['agent']], word_list[WORD_COLS[curr_voice]['verb']],
                                          word_list[WORD_COLS[curr_voice]['patient']]])
            else:
                valid_inds.append(i_word_list)
                content_words.append(
                    [word_list[WORD_COLS[curr_voice]['agent']], word_list[WORD_COLS[curr_voice]['verb']],
                     word_list[WORD_COLS[curr_voice]['patient']]])
        uni_content, labels = np.unique(np.array(content_words), axis=0, return_inverse=True)
        print(uni_content)
    else:
        labels = []
        valid_inds = []
        for i_sen_int, sen_int in enumerate(sen_ints):
            word_list = stimuli_voice[sen_int]['stimulus'].split()
            curr_voice = stimuli_voice[sen_int]['voice']
            if word == 'voice':
                labels.append(curr_voice)
                valid_inds.append(i_sen_int)
            elif word == 'senlen':
                if len(word_list) > 5:
                    labels.append('long')
                else:
                    labels.append('short')
                valid_inds.append(i_sen_int)
            elif word == 'agent' or word == 'patient':
                if experiment == 'PassAct3':
                    if len(word_list) > 5:
                        valid_inds.append(i_sen_int)
                        labels.append(word_list[WORD_COLS[curr_voice][word]])
                else:
                    labels.append(word_list[WORD_COLS[curr_voice][word]])
                    valid_inds.append(i_sen_int)
            else:
                labels.append(word_list[WORD_COLS[curr_voice][word]])
                valid_inds.append(i_sen_int)

    valid_inds = np.array(valid_inds)
    data = data[valid_inds, ...]
    sen_ints = [sen for i_sen, sen in enumerate(sen_ints) if i_sen in valid_inds]

    print(valid_inds)
    print(data.shape)
    print(sen_ints)
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

    if mode == 'acc':
        l_ints, cv_membership, tgm_acc, tgm_pred = models.lr_tgm_loso(data,
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
                            tgm_pred=tgm_pred,
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
    parser.add_argument('--mode', choices=['acc', 'coef'])

    args = parser.parse_args()
    print(args)
    # Check that parameter setting is valid
    total_valid = True
    is_valid = args.subject in VALID_SUBS[args.experiment]
    total_valid = total_valid and is_valid
    if not is_valid:
        print('subject wrong')
    if args.mode == 'coef':
        is_valid = not str_to_bool(args.doTestAvg)
        total_valid = total_valid and is_valid
        if not is_valid:
            print('Test avg makes no difference for coef mode')
    if args.word == 'voice':
        is_valid = args.sen_type == 'pooled'
        total_valid = total_valid and is_valid
        if not is_valid:
            print('Voice task only valid if sen_type == pooled')

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
                    proc=args.proc,
                    random_state_perm=args.perm_random_state,
                    force=str_to_bool(args.force),
                    mode=args.mode)
    else:
        print('Experiment parameters not valid. Skipping job.')
