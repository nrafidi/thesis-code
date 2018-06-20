import argparse
from syntax_vs_semantics import load_data
import models
import numpy as np
import os.path
import random
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_FILE = '{dir}TGM-LOSO_multisub_long_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'
NUM_SENTENCES = 16
NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']}
VALID_ALGS = ['lr-l2', 'lr-l1']
VALID_SEN_TYPE = ['active', 'passive']

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

TIME_LIMITS = {'PassAct3':
    {'active': {
            'noun1': {'tmin': -0.5, 'tmax': 6.0},
            'verb': {'tmin': 0.0, 'tmax': 3.5},
            'noun2': {'tmin': -1.0, 'tmax': 2.5}},
        'passive': {
            'noun1': {'tmin': -0.5, 'tmax': 6.0},
            'verb': {'tmin': -0.5, 'tmax': 3.0},
            'noun2': {'tmin': -1.0, 'tmax': 2.5}}},
'krns2': {
        'active': {
            'noun1': {'tmin': -0.5, 'tmax': 3.8},
            'verb': {'tmin': -1.0, 'tmax': 3.3},
            'noun2': {'tmin': -2.0, 'tmax': 2.3}},
        'passive': {
            'noun1': {'tmin': -0.5, 'tmax': 3.8},
            'verb': {'tmin': -1.5, 'tmax': 2.8},
            'noun2': {'tmin': -3.0, 'tmax': 1.3}}}}



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
                proc=load_data.DEFAULT_PROC,
                random_state_perm=1,
                force=False):
    warnings.filterwarnings(action='ignore')
    # Save Directory
    top_dir = TOP_DIR.format(exp=experiment)
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)

    fname = SAVE_FILE.format(dir=top_dir,
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
                             fold=fold)

    print(force)
    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return

    stimuli_voice = list(load_data.read_stimuli(experiment))

    data_list = []
    sen_ints = []
    time = []
    labels = []
    for i_sub, subject in enumerate(VALID_SUBS[experiment]):
        data, _, sen_ints_sub, time_sub, sensor_regions = load_data.load_sentence_data_v2(subject=subject,
                                                                                       align_to='noun1',
                                                                                       voice=sen_type,
                                                                                       experiment=experiment,
                                                                                       proc=proc,
                                                                                       num_instances=num_instances,
                                                                                       reps_filter=lambda x: [i for i in range(x) if i < 10],
                                                                                       sensor_type=None,
                                                                                       is_region_sorted=False,
                                                                                       tmin=TIME_LIMITS[experiment][sen_type]['noun1']['tmin'],
                                                                                       tmax=TIME_LIMITS[experiment][sen_type]['noun1']['tmax'])

        # print(labels_sub)
        # print(data.shape)
        valid_inds = []
        labels_sub = []
        new_sen_ints_sub = []
        for i_sen_int, sen_int in enumerate(sen_ints_sub):
            word_list = stimuli_voice[sen_int]['stimulus'].split()
            if word == 'noun2':
                if len(word_list) > 5:
                    labels_sub.append(word_list[WORD_COLS[sen_type][word]])
                    valid_inds.append(i_sen_int)
                    new_sen_ints_sub.append(sen_int)
            else:
                labels_sub.append(word_list[WORD_COLS[sen_type][word]])
                valid_inds.append(i_sen_int)
                new_sen_ints_sub.append(sen_int)

        valid_inds = np.array(valid_inds)
        print(labels_sub)
        data_list.append(data[valid_inds, ...])
        print(data_list[i_sub].shape)
        if i_sub == 0:
            sen_ints = new_sen_ints_sub
            time = time_sub
            labels = labels_sub
        else:
            assert np.all(sen_ints == new_sen_ints_sub)
            assert np.all(time == time_sub)
            assert np.all(np.array(labels) == np.array(labels_sub))

    tmin = time.min()
    tmax = time.max()

    total_win = int((tmax - tmin) * 500)

    if win_len < 0:
        win_len = total_win - overlap

    win_starts = range(0, total_win - win_len, overlap)
    # print(win_starts)
    print(sen_ints)
    print(labels)
    print(data_list[0].shape)


    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)

    l_ints, cv_membership, tgm_acc, tgm_pred = models.lr_tgm_loso_multisub_fold(data_list,
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
    parser.add_argument('--sen_type', choices=VALID_SEN_TYPE)
    parser.add_argument('--word', choices=['noun1', 'noun2', 'verb'])
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--isPerm', default='False', choices=['True', 'False'])
    parser.add_argument('--alg', default='lr-l2', choices=VALID_ALGS)
    parser.add_argument('--adj', default='zscore')
    parser.add_argument('--doTimeAvg', default='True', choices=['True', 'False'])
    parser.add_argument('--doTestAvg', default='True', choices=['True', 'False'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--force') #, default='False', choices=['True', 'False'])
    parser.add_argument('--fold', type=int)

    args = parser.parse_args()
    print(args)
    # Check that parameter setting is valid

    run_tgm_exp(experiment=args.experiment,
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
                proc=args.proc,
                random_state_perm=args.perm_random_state,
                force=str_to_bool(args.force))

