from syntax_vs_semantics import load_data
import numpy as np
import timeit

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


def str_to_none(str_thing):
    if str_thing =='None':
        return None
    else:
        return str_thing


# Runs the TGM experiment
def run_tgm_exp(data,
                labels,
                sen_ints,
                win_len,
                alg,
                doTimeAvg=False,
                doTestAvg=False):
    import models

    if 'l2' in alg:
        adj='zscore'
    else:
        adj=None

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


if __name__ == '__main__':

    data, _, sen_ints, time, _ = load_data.load_sentence_data_v2(subject='B',
                                                                 align_to='last',
                                                                 voice=['active', 'passive'],
                                                                 experiment='krns2',
                                                                 proc=load_data.DEFAULT_PROC,
                                                                 num_instances=1,
                                                                 reps_filter=None,
                                                                 sensor_type=None,
                                                                 is_region_sorted=False,
                                                                 tmin=0.5,
                                                                 tmax=1.0)

    stimuli_voice = list(load_data.read_stimuli('krns2'))
    labels = []
    for i_sen_int, sen_int in enumerate(sen_ints):
        curr_voice = stimuli_voice[sen_int]['voice']
        labels.append(curr_voice)

    print(labels)
    tmin = time.min()
    tmax = time.max()

    total_win = int((tmax - tmin) * 500)
    win_len = 25
    overlap = 12

    win_starts = range(0, total_win - win_len, overlap)

    all_times = []
    min_times = []
    for alg in VALID_ALGS:
        print(alg)
        t = timeit.Timer(stmt='run_tgm_exp(data, labels, sen_ints, win_len, "{alg}")'.format(alg=alg),
                         setup='from __main__ import run_tgm_exp, data, labels, sen_ints, win_len')
        times = t.repeat(repeat=10, number=1)
        times = np.array(times)
        all_times.append(times[None, ...])
        min_time = np.min(times)
        min_times.append(min_time)
        print(alg + ': %.2f' % min_time)

    all_times = np.concatenate(all_times, axis=0)
    min_times = np.array(min_times)
    np.savez('/share/volume0/nrafidi/alg_times_new.npz', all_times=all_times, min_times=min_times, algs=VALID_ALGS)
