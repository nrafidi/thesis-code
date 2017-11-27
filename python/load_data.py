import numpy as np
import hippo.io
import hippo.query
import h5py
import re

NUM_SENTENCES = 16
NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']}

WORD_INDS = {'firstNoun': 0, 'verb': 1, 'secondNoun': 2}

WORD_POS = {'active':
                {'firstNoun': 1,
                 'verb': 2,
                 'secondNoun': 4},
            'passive':
                {'firstNoun': 1,
                 'verb': 3,
                 'secondNoun': 6}}

USI_NAME = {'krns2': 'krns2',
            'PassAct2': 'pass-act-2',
            'PassAct3': 'pass-act-3'}

SEN_ID_RANGE = {'krns2':
                    {'active': range(4, 20),
                    'passive': range(20, 36)},
                'PassAct2':
                    {'active': range(16),
                    'passive': range(16, 32)},
                'PassAct3':
                    {'active': range(16),
                    'passive': range(16, 32)}}

ARTICLE1 = {'a': [4, 6, 9, 10, 13, 15, 16, 19, 21, 22, 24, 26, 28, 31, 33, 35],
            'the': [5, 7, 8, 11, 12, 14, 17, 18, 20, 23, 25, 27, 29, 30, 32, 34]}

ARTICLE2 = {'a': [5, 6, 8, 10, 12, 15, 17, 19, 20, 22, 25, 26, 29, 31, 32, 35],
            'the': [4, 7, 9, 11, 13, 14, 16, 18, 21, 23, 24, 27, 28, 30, 33, 34]}


WORD_PER_SEN = {'krns2':
                    {'active':
                         {'firstNoun': ['dog','doctor','student','monkey'],
                          'verb': ['found', 'kicked','inspected','touched'],
                          'secondNoun': ['peach.','school.','hammer.','door.']},
                     'passive':
                         {'firstNoun': ['peach','hammer','school','door'],
                          'verb': ['found', 'kicked', 'inspected', 'touched'],
                          'secondNoun': ['dog.', 'doctor.', 'student.', 'monkey.']}},
                'PassAct2':
                    {'active':
                         {'firstNoun': ['man', 'girl', 'woman', 'boy'],
                          'verb': ['watched', 'liked', 'despised', 'encouraged'],
                          'secondNoun': ['man.', 'girl.', 'woman.', 'boy.']},
                     'passive':
                         {'firstNoun': ['man', 'girl', 'woman', 'boy'],
                          'verb': ['watched', 'liked', 'despised', 'encouraged'],
                          'secondNoun': ['man.', 'girl.', 'woman.', 'boy.']}},
                'PassAct3':
                    {'active':
                         {'firstNoun': ['man', 'girl', 'woman', 'boy'],
                          'verb': ['kicked', 'helped', 'approached', 'punched'],
                          'secondNoun': ['man.', 'girl.', 'woman.', 'boy.']},
                     'passive':
                         {'firstNoun': ['man', 'girl', 'woman', 'boy'],
                          'verb': ['kicked', 'helped', 'approached', 'punched'],
                          'secondNoun': ['man.', 'girl.', 'woman.', 'boy.']}}}

TIME_LIMITS = {'active': {'firstNoun': {'tmin': -0.5, 'tmax': 4.0},
                          'verb': {'tmin': -1.0, 'tmax': 3.5},
                          'secondNoun': {'tmin': -2.0, 'tmax': 2.5}},
               'passive': {'firstNoun': {'tmin': -0.5, 'tmax': 4.0},
                           'verb': {'tmin': -1.5, 'tmax': 3.0},
                           'secondNoun': {'tmin': -3.0, 'tmax': 1.5}}}

# Old slugs:
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'sss_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-5-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_lp-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
DEFAULT_PROC = 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_lp-150_notch-60-120_beatremoval-first_blinkremoval-first'
PDTW_FILE = '/share/volume0/newmeg/{exp}/avg/{exp}_{sub}_{proc}_parsed_{word}_pdtwSyn.mat'
GLOVE = '/share/volume1/sjat/repositories/Nicole-thesis-code/python/sj-experiments/embeddings_files/glove.840B.300d.txt'


def get_sen_num_from_id(sen_id):
    m = re.match('.*sentence-(\d+)', sen_id)
    return int(m.group(1))


def load_glove_vectors(labels):
    words = [str(w.replace('.','')) for w in labels]

    vec_dict = {}
    with open(GLOVE) as lines:
        for line in lines:
            split = line.split(" ")
            word = str(split[0])

            vector_strings = split[1:]
            vector = [float(num) for num in vector_strings]
            if word in words:
                vec_dict[word] = np.array(vector)
    vec = np.empty((len(labels), 300))
    for i_w, w in enumerate(words):
        vec[i_w, :] = vec_dict[w]
    return vec


def get_arts_from_senid(sentence_ids, art_num):
    arts = []
    for sen in sentence_ids:
        if art_num == 1:
            for key in ARTICLE1:
                if sen in ARTICLE1[key]:
                    arts.append(key)
        elif art_num ==2:
            for key in ARTICLE2:
                if sen in ARTICLE2[key]:
                    arts.append(key)
        else:
            raise NameError('There are only two articles, dingus')
    if len(arts) != len(sentence_ids):
        raise AssertionError('Should be same length:\narts={}\nsentence_ids={}'.format(arts, sentence_ids))
    return arts


def load_one_hot(labels):
    unique_labels = np.unique(labels)
    num_labels = unique_labels.size
    one_hot_dict = {}
    for i_l, uni_l in enumerate(unique_labels):
        one_hot_dict[uni_l] = np.zeros((num_labels,))
        one_hot_dict[uni_l][i_l] = 1
    one_hot = []
    for l in labels:
        one_hot.append(one_hot_dict[l])
    return np.stack(one_hot)


def load_pdtw(subject, word, experiment='krns2', proc=DEFAULT_PROC):
    fname = PDTW_FILE.format(exp=experiment, sub=subject, proc=proc, word=word)
    loadVars = h5py.File(fname)

    time_a = np.array(loadVars[u'fullTime_a'])
    time_p = np.array(loadVars[u'fullTime_p'])

    fullWordLab = loadVars[u'fullWordLab']
    labels = fullWordLab[WORD_INDS[word], :]

    active_data_raw = np.transpose(loadVars[u'activeData'], axes=(2, 1, 0))
    passive_data_raw = np.transpose(loadVars[u'passiveData'], axes=(2, 1, 0))

    return time_a, time_p, labels, active_data_raw, passive_data_raw


def load_raw(subject, word, sen_type, experiment='krns2', proc=DEFAULT_PROC,
             tmin=None, tmax=None):
    usis = hippo.query.query_usis([('stimuli_set', USI_NAME[experiment]),
                                   ('stimulus', lambda s: s in WORD_PER_SEN[experiment][sen_type][word]),
                                   # without periods, gets the first noun
                                   ('sentence_id', lambda sid: sid != None and (get_sen_num_from_id(sid) in SEN_ID_RANGE[experiment][sen_type])),
                                   ('word_index_in_sentence', lambda wis: wis == WORD_POS[sen_type][word])],
                                  include_annotations=['stimulus', 'sentence_id'])  # excludes questions
    exp_sub = [(experiment, subject)]
    uels = hippo.query.get_uels_from_usis(usis.keys(), experiment_subjects=exp_sub)
    uels = {k: v for (k, v) in uels.iteritems() if len(v) > 0}  # checking for empties
    id_uels = [(k, uels[k]) for k in uels.keys()]  # putting uels in a list instead of a map (explicit ordering)
    labels = [usis[k]['stimulus'] for k, _ in id_uels]
    sentence_ids = [usis[k]['sentence_id'] for k, _ in id_uels]

    _, uels = zip(*id_uels)

    if tmin == None:
        tmin = TIME_LIMITS[sen_type][word]['tmin']
    if tmax == None:
        tmax = TIME_LIMITS[sen_type][word]['tmax']

    evokeds = np.array([hippo.io.load_mne_epochs(us, preprocessing=proc, baseline=None,
                                        tmin=tmin, tmax=tmax) for us in uels])

    # Downsample
    evokeds = evokeds[:, :, :, ::2]
    time = np.arange(tmin, tmax+2e-3, 2e-3)
    print(evokeds.shape[3])
    print(time.size)
    if evokeds.shape[3] != time.size:
        min_size = np.min([evokeds.shape[3], time.size])
        evokeds = evokeds[:, :, :, :min_size]
        time = time[:min_size]

    return evokeds, labels, time, sentence_ids


def avg_data(data_raw, labels_raw, sentence_ids_raw=None, experiment='krns2', num_instances=2, reps_to_use=-1):
    num_reps = NUM_REPS[experiment]
    if reps_to_use == -1:
        reps_to_use = num_reps

    if len(data_raw.shape) > 3:
        data_list = []
        for i in range(num_instances):
            data_list.append(np.mean(data_raw[:, range(i, reps_to_use, num_instances), :, :], axis=1))
        data = np.concatenate(data_list, axis=0)
    else:
        data = np.empty((NUM_SENTENCES * num_instances, data_raw.shape[1], data_raw.shape[2]))
        for s in range(NUM_SENTENCES):
            startInd = num_reps * s
            endInd = startInd + reps_to_use - 1
            for i in range(num_instances):
                data[s + (NUM_SENTENCES*i), :, :] = np.mean(data_raw[(startInd + i):endInd:num_instances, :, :], axis=0)

    labels = []
    sentence_ids = []
    for i in range(num_instances):
        labels.extend(labels_raw)
        if sentence_ids_raw is not None:
            sentence_ids.extend(sentence_ids_raw)

    return data, labels, sentence_ids


