import argparse
from syntax_vs_semantics import load_data
import numpy as np
import os.path
import random
import warnings
from scipy.spatial.distance import pdist, squareform

TOP_DIR = '/share/volume0/nrafidi/{exp}_RSA/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}RSA_{sub}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}'

NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']}
VALID_ALGS = ['lr-l2', 'lr-l1']
VALID_SEN_TYPE = ['active', 'passive', 'pooled']

TMIN={'det': -0.5,
      'noun2': 0.0}
TMAX={'det': 0.0,
      'noun2': 0.5}

WORD_COLS = {'active': {'det': 3, 'noun2': 4},
             'passive': {'det': 5, 'noun2': 6}}


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


def my_cosine(vec1, vec2):
    vec1_orig = vec1
    vec2_orig = vec2
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    if np.isnan(np.dot(vec1, vec2)):
        print('norms:')
        print(norm1)
        print(np.min(vec1_orig))
        print(norm2)
        print(np.min(vec2_orig))
    return 1.0 - np.dot(vec1, vec2)


# Runs the TGM experiment
def run_tgm_exp(experiment,
                subject,
                word,
                win_len,
                overlap,
                dist='cosine',
                doTimeAvg=False,
                proc=load_data.DEFAULT_PROC,
                force=False):

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
                             dist=dist,
                             avgTm=bool_to_str(doTimeAvg))

    print(force)
    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return

    voice = ['active', 'passive']
    num_instances= 1

    all_data, _, sen_ints, time, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                     align_to='last',
                                                                     voice=voice,
                                                                     experiment=experiment,
                                                                     proc=proc,
                                                                     num_instances=num_instances,
                                                                     reps_filter=None,
                                                                     sensor_type=None,
                                                                     is_region_sorted=False,
                                                                     tmin=TMIN[word],
                                                                     tmax=TMAX[word])
    all_data *= 1e12
    print(np.min(all_data))
    print(np.min(np.abs(all_data)))
    print(np.any(np.isnan(all_data)))
    print(np.any(np.isinf(all_data)))
    print(all_data.shape)
    stimuli_voice = list(load_data.read_stimuli(experiment))
    labels = []
    voice_labels = []
    data = np.ones((all_data.shape[0]/2, all_data.shape[1], all_data.shape[2]))
    print(data.shape)
    i_data = 0
    for i_sen_int, sen_int in enumerate(sen_ints):
        word_list = stimuli_voice[sen_int]['stimulus'].split()
        curr_voice = stimuli_voice[sen_int]['voice']
        if len(word_list) > 5:
            data[i_data, :, :] = all_data[i_sen_int, :, :]
            labels.append(word_list[WORD_COLS[curr_voice][word]])
            voice_labels.append(curr_voice)
        i_data += 1
    print(labels)
    print(voice_labels)
    tmin = time.min()
    tmax = time.max()

    total_win = int((tmax - tmin) * 500)

    if win_len < 0:
        win_len = total_win - overlap

    win_starts = range(0, total_win - win_len, overlap)
    n_time = data.shape[2]
    windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
    n_w = len(windows)

    RDM = []
    for wi in xrange(n_w):
        time_to_use = windows[wi]
        data_to_use = data[:, :, time_to_use]
        if doTimeAvg:
            data_to_use = np.mean(data_to_use, axis=2)
        else:
            data_to_use = np.reshape(data_to_use, (data_to_use.shape[0], -1))
        curr_RDM = squareform(pdist(data_to_use, metric=dist))
        if np.any(np.isnan(curr_RDM)):
            print('Data state:')
            print(np.any(np.isinf(data_to_use)))
            print(np.any(np.isnan(data_to_use)))
            print(np.min(data_to_use))
            print(np.min(np.abs(data_to_use)))
            meow = pdist(data_to_use, metric=my_cosine)
            nan_els = np.unravel_index(np.where(np.isnan(meow)), curr_RDM.shape)
            # print(nan_els)
            print('My cosine:')
            print my_cosine(data_to_use[nan_els[0][0][0], :], data_to_use[nan_els[1][0][0], :])
        RDM.append(curr_RDM[None, ...])

    RDM = np.concatenate(RDM, axis=0)
    np.savez_compressed(fname,
                        RDM=RDM,
                        labels=labels,
                        voice_labels=voice_labels,
                        win_starts=win_starts,
                        time=time,
                        proc=proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--word', choices=['det', 'noun2'])
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='False', choices=['True', 'False'])
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--force', default='False', choices=['True', 'False'])

    args = parser.parse_args()
    print(args)

    warnings.filterwarnings(action='ignore')
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
                    dist=args.dist,
                    doTimeAvg=str_to_bool(args.doTimeAvg),
                    proc=args.proc,
                    force=str_to_bool(args.force))
    else:
        print('Experiment parameters not valid. Skipping job.')
