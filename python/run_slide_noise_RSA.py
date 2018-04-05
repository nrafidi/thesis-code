import argparse
from syntax_vs_semantics import load_data
import numpy as np
import os.path
import itertools
import warnings
from scipy.spatial.distance import pdist, squareform

TOP_DIR = '/share/volume0/nrafidi/{exp}_RSA/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}RSA_{sub}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_{draw}'

NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']}
VALID_ALGS = ['lr-l2', 'lr-l1']
VALID_SEN_TYPE = ['active', 'passive', 'pooled']

TMIN={'det': -0.5,
      'det-full': -0.5,
      'noun2': 0.0,
      'last-full': 0.0,
      'eos': 0.5,
      'eos-full': 0.5}
TMAX={'det': 0.0,
      'det-full': 0.0,
      'noun2': 0.5,
      'last-full': 0.5,
      'eos': 1.0,
      'eos-full': 1.0}

WORD_COLS = {'active': {'det': 3, 'det-full': -3, 'noun2': 4, 'last-full': -2, 'eos': 4, 'eos-full': -2},
             'passive': {'det': 5, 'det-full': -3, 'noun2': 6, 'last-full': -2, 'eos': 6, 'eos-full': -2}}


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


def load_agg_data(subject, word, experiment, voice, proc, rep_set):
    all_data, _, sen_ints, time, _ = load_data.load_sentence_data_v2(subject=subject,
                                                                     align_to='last',
                                                                     voice=voice,
                                                                     experiment=experiment,
                                                                     proc=proc,
                                                                     num_instances=1,
                                                                     reps_filter=lambda rep_list: [
                                                                         rep in rep_set
                                                                         for rep in rep_list],
                                                                     sensor_type=None,
                                                                     is_region_sorted=False,
                                                                     tmin=TMIN[word],
                                                                     tmax=TMAX[word])
    all_data *= 1e12
    stimuli_voice = list(load_data.read_stimuli(experiment))
    labels = []
    voice_labels = []
    if 'full' not in word:
        data = np.ones((all_data.shape[0] / 2, all_data.shape[1], all_data.shape[2]))
    else:
        data = all_data
    i_data = 0
    for i_sen_int, sen_int in enumerate(sen_ints):
        word_list = stimuli_voice[sen_int]['stimulus'].split()
        curr_voice = stimuli_voice[sen_int]['voice']
        if 'full' in word:
            labels.append(word_list[-2])
            voice_labels.append(curr_voice)
        elif len(word_list) > 5:
            data[i_data, :, :] = all_data[i_sen_int, :, :]
            labels.append(word_list[WORD_COLS[curr_voice][word]])
            voice_labels.append(curr_voice)
            i_data += 1
    print(labels)
    print(voice_labels)
    return data, labels, voice_labels, time


def make_rdm(data, windows, dist, doTimeAvg):
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
        assert not np.any(np.isnan(curr_RDM))
        RDM.append(curr_RDM[None, ...])

    return np.concatenate(RDM, axis=0)


# Runs the RSA experiment
def run_rsa_exp(experiment,
                subject,
                word,
                win_len,
                overlap,
                draw,
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
                             avgTm=bool_to_str(doTimeAvg),
                             draw=draw)

    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return

    voice = ['active', 'passive']
    num_reps = NUM_REPS[experiment]
    rep_draws = list(itertools.combinations(range(num_reps), num_reps/2))
    val_reps = list(rep_draws[draw])
    val_data, val_labels, val_voice_labels, val_time = load_agg_data(subject, word, experiment, voice, proc, val_reps)
    test_reps = list(rep_draws[len(rep_draws) - draw])
    test_data, test_labels, test_voice_labels, test_time = load_agg_data(subject, word, experiment, voice, proc, test_reps)

    assert np.all(np.array(val_labels) == np.array(test_labels))
    assert np.all(np.array(val_voice_labels) == np.array(test_voice_labels))
    assert np.all(np.array(val_time) == np.array(test_time))

    time = val_time
    labels = val_labels
    voice_labels = val_voice_labels

    tmin = time.min()
    tmax = time.max()

    total_win = int((tmax - tmin) * 500)

    if win_len < 0:
        win_len = total_win - overlap

    win_starts = range(0, total_win - win_len, overlap)
    n_time = val_data.shape[2]
    windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]

    val_rdm = make_rdm(val_data, windows, dist, doTimeAvg)
    test_rdm = make_rdm(test_data, windows, dist, doTimeAvg)

    np.savez_compressed(fname,
                        val_rdm=val_rdm,
                        test_rdm=test_rdm,
                        labels=labels,
                        voice_labels=voice_labels,
                        win_starts=win_starts,
                        time=time,
                        proc=proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--word', choices=TMIN.keys())
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--draw', type=int)
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
        run_rsa_exp(experiment=args.experiment,
                    subject=args.subject,
                    word=args.word,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    draw=args.draw,
                    dist=args.dist,
                    doTimeAvg=str_to_bool(args.doTimeAvg),
                    proc=args.proc,
                    force=str_to_bool(args.force))
    else:
        print('Experiment parameters not valid. Skipping job.')
