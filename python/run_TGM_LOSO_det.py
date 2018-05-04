import argparse
from syntax_vs_semantics import load_data
import models
import numpy as np
import os.path
import random
import warnings

TOP_DIR = '/share/volume0/nrafidi/krns2_TGM_LOSO_det/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO-det_{sub}_{sen_type}_{analysis}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'

VALID_SUBS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
VALID_ALGS = ['lr-l2', 'lr-l1']
VALID_SEN_TYPE = ['pooled', 'active', 'passive']
VALID_ANALYSES = ['det-type', 'the-dog']

WORD_COLS = {'active': {'det1': 0,
                        'noun1': 1,
                        'verb': 2,
                        'det2': 3,
                        'noun2': 4,
                        'agent': 1,
                        'patient': 4},
             'passive': {'det1': 0,
                         'noun1': 1,
                         'verb': 3,
                         'det2': 5,
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

# Runs the TGM experiment
def run_tgm_exp(subject,
                sen_type,
                analysis,
                win_len,
                overlap,
                isPerm = False,
                alg='lr-l1',
                adj=None,
                doTimeAvg=False,
                doTestAvg=True,
                num_instances=1,
                random_state_perm=1,
                force=False,
                mode='acc'):
    warnings.filterwarnings(action='ignore')
    # Save Directory
    if not os.path.exists(TOP_DIR):
        os.mkdir(TOP_DIR)
    save_dir = SAVE_DIR.format(top_dir=TOP_DIR, sub=subject)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname = SAVE_FILE.format(dir=save_dir,
                             sub=subject,
                             sen_type=sen_type,
                             analysis=analysis,
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
        voice = sen_type
    experiment = 'krns2'

    data_det1, _, sen_ints_det1, _, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                        align_to='noun1',
                                                                        voice=voice,
                                                                        experiment=experiment,
                                                                        proc=load_data.DEFAULT_PROC,
                                                                        num_instances=num_instances,
                                                                        reps_filter=None,
                                                                        sensor_type=None,
                                                                        is_region_sorted=False,
                                                                        tmin=-0.5,
                                                                        tmax=0.0)

    data_n1, _, sen_ints_n1, _, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                    align_to='noun1',
                                                                    voice=voice,
                                                                    experiment=experiment,
                                                                    proc=load_data.DEFAULT_PROC,
                                                                    num_instances=num_instances,
                                                                    reps_filter=None,
                                                                    sensor_type=None,
                                                                    is_region_sorted=False,
                                                                    tmin=0.0,
                                                                    tmax=0.5)

    data_det2, _, sen_ints_det2, _, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                        align_to='noun2',
                                                                        voice=voice,
                                                                        experiment=experiment,
                                                                        proc=load_data.DEFAULT_PROC,
                                                                        num_instances=num_instances,
                                                                        reps_filter=None,
                                                                        sensor_type=None,
                                                                        is_region_sorted=False,
                                                                        tmin=-0.5,
                                                                        tmax=0.0)

    data_n2, _, sen_ints_n2, time, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                       align_to='noun2',
                                                                        voice=voice,
                                                                        experiment=experiment,
                                                                        proc=load_data.DEFAULT_PROC,
                                                                        num_instances=num_instances,
                                                                        reps_filter=None,
                                                                        sensor_type=None,
                                                                        is_region_sorted=False,
                                                                        tmin=0.0,
                                                                        tmax=0.5)

    stimuli_voice = list(load_data.read_stimuli(experiment))
    labels = []
    for sen_int in sen_ints_det1:
        word_list = stimuli_voice[sen_int]['stimulus'].split()
        curr_voice = stimuli_voice[sen_int]['voice']
        labels.append(word_list[WORD_COLS[curr_voice]['det1']])
    for sen_int in sen_ints_n1:
        word_list = stimuli_voice[sen_int]['stimulus'].split()
        curr_voice = stimuli_voice[sen_int]['voice']
        labels.append(word_list[WORD_COLS[curr_voice]['noun1']])
    for sen_int in sen_ints_det2:
        word_list = stimuli_voice[sen_int]['stimulus'].split()
        curr_voice = stimuli_voice[sen_int]['voice']
        labels.append(word_list[WORD_COLS[curr_voice]['det2']])
    for sen_int in sen_ints_n2:
        word_list = stimuli_voice[sen_int]['stimulus'].split()
        curr_voice = stimuli_voice[sen_int]['voice']
        labels.append(word_list[WORD_COLS[curr_voice]['noun2']])

    sen_ints = np.concatenate([sen_ints_det1, sen_ints_n1, sen_ints_det2, sen_ints_n2], axis=0)
    data = np.concatenate([data_det1, data_n1, data_det2, data_n2], axis=0)

    inds_to_keep = np.ones((len(labels),), dtype=bool)
    if analysis == 'the-dog':
        for i_label, label in enumerate(labels):
            if label != 'the' and label != 'dog':
                inds_to_keep[i_label] = False

    data = data[inds_to_keep, :, :]
    sen_ints = sen_ints[inds_to_keep]
    new_labels = [labels[i_label] for i_label, _ in enumerate(labels) if inds_to_keep[i_label]]

    total_win = data.shape[-1]
    win_starts = range(0, total_win - win_len, overlap)

    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)

    if mode == 'acc':
        l_ints, cv_membership, tgm_acc, tgm_pred = models.lr_tgm_loso(data,
                                                                      new_labels,
                                                                      win_starts,
                                                                      win_len,
                                                                      sen_ints,
                                                                      penalty=alg[3:],
                                                                      adj=adj,
                                                                      doTimeAvg=doTimeAvg,
                                                                      doTestAvg=doTestAvg)
        print(np.mean(tgm_acc, axis=0))
        np.savez_compressed(fname,
                            l_ints=l_ints,
                            cv_membership=cv_membership,
                            tgm_acc=tgm_acc,
                            tgm_pred=tgm_pred,
                            win_starts=win_starts,
                            time=time)
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
                            time=time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject')
    parser.add_argument('--sen_type', choices=VALID_SEN_TYPE)
    parser.add_argument('--analysis', choices=VALID_ANALYSES)
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--isPerm', default='False', choices=['True', 'False'])
    parser.add_argument('--alg', default='lr-l2', choices=VALID_ALGS)
    parser.add_argument('--adj', default='zscore')
    parser.add_argument('--doTimeAvg', default='False', choices=['True', 'False'])
    parser.add_argument('--doTestAvg', default='True', choices=['True', 'False'])
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--force', default='False', choices=['True', 'False'])
    parser.add_argument('--mode', default='acc', choices=['acc', 'coef'])

    args = parser.parse_args()
    print(args)
    # Check that parameter setting is valid
    total_valid = True
    is_valid = args.subject in VALID_SUBS
    total_valid = total_valid and is_valid
    if not is_valid:
        print('subject wrong')
    if args.mode == 'coef':
        is_valid = not str_to_bool(args.doTestAvg)
        total_valid = total_valid and is_valid
        if not is_valid:
            print('Test avg makes no difference for coef mode')

    if total_valid:
        run_tgm_exp(subject=args.subject,
                    sen_type=args.sen_type,
                    analysis=args.analysis,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    isPerm=str_to_bool(args.isPerm),
                    alg=args.alg,
                    adj=args.adj,
                    doTimeAvg=str_to_bool(args.doTimeAvg),
                    doTestAvg=str_to_bool(args.doTestAvg),
                    num_instances=args.num_instances,
                    random_state_perm=args.perm_random_state,
                    force=str_to_bool(args.force),
                    mode=args.mode)
    else:
        print('Experiment parameters not valid. Skipping job.')
