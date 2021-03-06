import numpy as np
import hippo.io
import hippo.query
import re
import string
import path_constants

NUM_SENTENCES = 16
NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']}

WORD_INDS = {'noun1': 0, 'verb': 1, 'noun2': 2}

EXP_INDS = {'krns2': range(64, 96),
            'PassAct2': range(32, 64),
            'PassAct3': range(32)}

WORD_POS = {'active':
                {'noun1': 1,
                 'verb': 2,
                 'noun2': 4,
                 'agent': 1,
                 'patient': 4},
            'passive':
                {'noun1': 1,
                 'verb': 3,
                 'noun2': 6,
                 'agent': 6,
                 'patient': 1}}

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
                    {'active': [0, 1, 2, 3, 8, 9, 10, 11],
                     'passive': [16, 17, 18, 19, 24, 25, 26, 27]}}

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


def load_raw(subject, word, sen_type, experiment='krns2', proc=DEFAULT_PROC,
             tmin=None, tmax=None):
    if experiment == 'PassAct3':
        usis = hippo.query.query_usis([('stimuli_set', USI_NAME[experiment]),
                                       ('stimulus', lambda s: s in WORD_PER_SEN[experiment][sen_type][word]),
                                       # without periods, gets the first noun
                                       ('sentence_id', lambda sid: sid != None),
                                       ('word_index_in_sentence', lambda wis: wis == WORD_POS[sen_type][word])],
                                      include_annotations=['stimulus', 'sentence_id'])  # excludes questions
    else:
        usis = hippo.query.query_usis([('stimuli_set', USI_NAME[experiment]),
                                       ('stimulus', lambda s: s in WORD_PER_SEN[experiment][sen_type][word]),
                                       # without periods, gets the first noun
                                       ('sentence_id', lambda sid: sid != None and (
                                           get_sen_num_from_id(sid) in SEN_ID_RANGE[experiment][sen_type])),
                                       ('word_index_in_sentence', lambda wis: wis == WORD_POS[sen_type][word])],
                                      include_annotations=['stimulus', 'sentence_id'])  # excludes questions
    print('{} usis loaded:'.format(len(usis.keys())))
    print(usis.keys())
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
    time = np.arange(tmin, tmax + 2e-3, 2e-3)
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
                data[s + (NUM_SENTENCES * i), :, :] = np.mean(data_raw[(startInd + i):endInd:num_instances, :, :],
                                                              axis=0)

    labels = []
    sentence_ids = []
    for i in range(num_instances):
        labels.extend(labels_raw)
        if sentence_ids_raw is not None:
            sentence_ids.extend(sentence_ids_raw)

    return data, labels, sentence_ids


noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
to_be_verbs = {'be', 'am', 'is', 'are', 'being', 'was', 'were', 'been'}
punctuation_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))


def has_part_of_speech(ordered_sentence_usis, allowed_pos):
    for usi, annotation in ordered_sentence_usis:
        yield ((usi, annotation), annotation['stanford_2017_06_09_pos'] in allowed_pos)


def is_non_to_be_verb_at_count(ordered_sentence_usis, index_non_to_be_verb, is_raise=True):
    index = 0
    for (usi, annotation), is_match in has_part_of_speech(ordered_sentence_usis, verb_tags):
        if is_match:
            lower_text = punctuation_regex.sub('', annotation['stimulus']).lower()
            if lower_text in to_be_verbs:
                is_match = False
        if is_match:
            if index == index_non_to_be_verb:
                yield True
            else:
                yield False
            index += 1
        else:
            yield False
    if is_raise and index == 0:
        raise ValueError('Not enough non-to-be-verbs found in stimulus')


def is_noun_at_noun_count(ordered_sentence_usis, index_noun, is_raise=True):
    index = 0
    for (usi, annotation), is_match in has_part_of_speech(ordered_sentence_usis, noun_tags):
        if is_match:
            if index == index_noun:
                yield True
            else:
                yield False
            index += 1
        else:
            yield False
    if is_raise and index == 0:
        raise ValueError('Not enough nouns found in stimulus')


def is_in_long_sentence(ordered_sentence_usis):
    count = 0
    result = False
    for usi, annotation in ordered_sentence_usis:
        count += 1
        if count > 4:
            result = True
    for _ in range(count):
        yield result


def is_first_noun(ordered_sentence_usis, is_raise=True):
    return is_noun_at_noun_count(ordered_sentence_usis, 0)


def is_second_noun(ordered_sentence_usis, is_raise=True):
    return is_noun_at_noun_count(ordered_sentence_usis, 1)


def is_first_non_to_be_verb(ordered_sentence_usis, is_raise=True):
    return is_non_to_be_verb_at_count(ordered_sentence_usis, 0)


def is_in_passive(ordered_sentence_usis):
    count = 0
    result = False
    for usi, annotation in ordered_sentence_usis:
        count += 1
        lower_text = punctuation_regex.sub('', annotation['stimulus']).lower()
        if lower_text == 'was':
            result = True
    for _ in range(count):
        yield result


def is_in_active(ordered_sentence_usis):
    for result in is_in_passive(ordered_sentence_usis):
        yield not result


def recon_sen_from_usis(usis):
    sentence_id_to_stimulus = dict()
    sentence_id_to_word_index = dict()
    sentence_id_to_usi = dict()
    for u in usis:
        sentence_id = usis[u]['sentence_id']
        if sentence_id is not None:
            if sentence_id not in sentence_id_to_stimulus.keys():
                sentence_id_to_stimulus[sentence_id] = []
                sentence_id_to_word_index[sentence_id] = []
                sentence_id_to_usi[sentence_id] = []
            sentence_id_to_stimulus[sentence_id].append(usis[u]['stimulus'])
            sentence_id_to_word_index[sentence_id].append(usis[u]['word_index_in_sentence'])
            sentence_id_to_usi[sentence_id].append(u)
    recon_sentences = []
    sentence_id_by_recon = []
    for sentence_id in sentence_id_to_stimulus.keys():
        word_indices = np.array(sentence_id_to_word_index[sentence_id])
        words = np.array(sentence_id_to_stimulus[sentence_id])
        for i_word, word in enumerate(words):
            if 'punched' in word:
                words[i_word] = 'unklced'
        corr_words = np.array([punctuation_regex.sub('', word).lower() for word in words])
        sorted_inds = np.argsort(word_indices)
        recon = ' '.join(corr_words[sorted_inds])
        recon_sentences.append(recon)
        sentence_id_by_recon.append(sentence_id)
    return recon_sentences, sentence_id_by_recon


def order_sentences(usis, experiment):
    recon_sentences, sentence_id_by_recon = recon_sen_from_usis(usis)
    print(recon_sentences)
    with open(path_constants.SENTENCES) as f:
        loaded_sentences = f.readlines()
    loaded_sentences = [sen.strip() for sen in loaded_sentences]
    exp_sentences = [punctuation_regex.sub('', loaded_sentences[ind]).lower().strip() for ind in EXP_INDS[experiment]]
    print(exp_sentences)
    sorted_inds = [recon_sentences.index(sen) for sen in exp_sentences]
    sorted_sentence_ids = [sentence_id_by_recon[ind] for ind in sorted_inds]
    test_sort = [recon_sentences[ind] for ind in sorted_inds]
    assert test_sort == exp_sentences
    return sorted_sentence_ids


def load_raw_v2(subject, experiment, filters, tmin, tmax, proc=None):
    # for example:
    # evoked, labels, sentence_ids, time = load_raw('A', 'PassAct3', [is_in_active, is_first_noun], tmin=-0.7, tmax=2.5)

    if proc is None:
        proc = 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_lp-150_notch-60-120_beatremoval-first_blinkremoval-first'

    usis = hippo.query.query_usis([('stimuli_set', USI_NAME[experiment])],
                                  include_annotations=[
                                      'stimulus',
                                      'sentence_id',
                                      'word_index_in_sentence',
                                      'stanford_2017_06_09_pos',
                                      'question_id'])

    # sort in text file sentence order
    sorted_sentence_ids = order_sentences(usis, experiment)
    # group by sentence ids
    sentence_id_to_usis = dict()
    for usi in usis:
        annotations = usis[usi]
        # filter out question words on this side, since None is not handled correctly by hippo
        if annotations['question_id'] is not None:
            continue
        sentence_id = annotations['sentence_id']
        if sentence_id in sentence_id_to_usis:
            sentence_id_to_usis[sentence_id].append((usi, annotations))
        else:
            sentence_id_to_usis[sentence_id] = [(usi, annotations)]

    filtered_usis = dict()
    for sentence_id in sentence_id_to_usis:
        sentence_usis = sentence_id_to_usis[sentence_id]
        # sorted order
        usi_words = sorted(sentence_usis, key=lambda usi_annotation: usi_annotation[1]['word_index_in_sentence'])
        anded_filter = [True for _ in range(len(usi_words))]
        for f in filters:
            for idx, result in enumerate(f(usi_words)):
                anded_filter[idx] = anded_filter[idx] and result
            assert (idx == len(anded_filter) - 1)  # if this is violated the filter is messed up

        for is_allowed, usi_word in zip(anded_filter, usi_words):
            # print(usi_word[1]['stimulus'], usi_word[1]['stanford_2017_06_09_pos'], is_allowed)
            if is_allowed:
                filtered_usis[usi_word[0]] = usi_word[1]

    usis = filtered_usis

    print(usis)
    print(len(usis))

    exp_sub = [(experiment, subject)]
    uels = hippo.query.get_uels_from_usis(usis.keys(), experiment_subjects=exp_sub)
    id_uels = [(k, uels[k]) for k in uels.keys()]  # putting uels in a list instead of a map (explicit ordering)
    labels = [punctuation_regex.sub('', usis[k]['stimulus']).lower() for k, _ in id_uels]
    sentence_ids = [usis[k]['sentence_id'] for k, _ in id_uels]
    assert len(labels) == len(sentence_ids)
    sorted_inds_sentence = [sentence_ids.index(sen_id) for sen_id in sorted_sentence_ids if sen_id in sentence_ids]
    print(sorted_inds_sentence)
    labels = [labels[ind] for ind in sorted_inds_sentence]
    print(labels)
    sentence_ids = [sentence_ids[ind] for ind in sorted_inds_sentence]

    _, uels = zip(*id_uels)

    evokeds = np.array(
        [hippo.io.load_mne_epochs(us, preprocessing=proc, baseline=None, tmin=tmin, tmax=tmax) for us in uels])

    evokeds = evokeds[np.array(sorted_inds_sentence), ...]
    # Downsample
    evokeds = evokeds[:, :, :, ::2]
    time = np.arange(tmin, tmax + 2e-3, 2e-3)
    print(evokeds.shape[3])
    print(time.size)
    if evokeds.shape[3] != time.size:
        min_size = np.min([evokeds.shape[3], time.size])
        evokeds = evokeds[:, :, :, :min_size]
        time = time[:min_size]

    return evokeds, labels, sentence_ids, time


def load_sentence_data(subject, word, sen_type, experiment, proc, num_instances, reps_to_use, sorted_inds=None):
    evokeds, labels, time, sen_ids = load_raw(subject, word, sen_type,
                                              experiment=experiment, proc=proc)

    data, labels, sen_ids = avg_data(evokeds, labels, sentence_ids_raw=sen_ids,
                                     experiment=experiment,
                                     num_instances=num_instances,
                                     reps_to_use=reps_to_use)

    labels = np.array(labels)
    label_sort_inds = np.argsort(sen_ids)  # TODO: check that sen_ids are ordered consistently with sentence list
    # More robust thing to do is to use strings and a dictionary and not have an unnecessary mapping step
    # Explore alternative querying of hippocampus - try querying with sentence id and see what stimulus is
    labels = labels[label_sort_inds]
    data = data[label_sort_inds, :, :]

    if sorted_inds is not None:
        data = data[:, sorted_inds, :]

    return data, labels, time


def load_sentence_data_v2(subject, word, sen_type, experiment, proc, num_instances, reps_to_use, noMag=False,
                          sorted_inds=None):
    # evoked, labels, sentence_ids, time = load_raw('A', 'PassAct3', [is_in_active, is_first_noun], tmin=-0.7, tmax=2.5)

    tmin = TIME_LIMITS[sen_type][word]['tmin']
    tmax = TIME_LIMITS[sen_type][word]['tmax']

    if sen_type == 'active':
        if word == 'noun1':
            filters = [is_in_long_sentence, is_in_active, is_first_noun]
        elif word == 'verb':
            filters = [is_in_long_sentence, is_in_active, is_first_non_to_be_verb]
        elif word == 'noun2':
            filters = [is_in_long_sentence, is_in_active, is_second_noun]
        elif word == 'agent':
            filters = [is_in_long_sentence, is_in_active, is_first_noun]
        else:
            filters = [is_in_long_sentence, is_in_active, is_second_noun]
    else:
        if word == 'noun1':
            filters = [is_in_long_sentence, is_in_passive, is_first_noun]
        elif word == 'verb':
            filters = [is_in_long_sentence, is_in_passive, is_first_non_to_be_verb]
        elif word == 'noun2':
            filters = [is_in_long_sentence, is_in_passive, is_second_noun]
        elif word == 'agent':
            filters = [is_in_long_sentence, is_in_passive, is_second_noun]
        else:
            filters = [is_in_long_sentence, is_in_passive, is_first_noun]

    evokeds, labels, time, sen_ids = load_raw_v2(subject, experiment, filters, tmin, tmax)

    data, labels, sen_ids = avg_data(
        evokeds, labels,
        sentence_ids_raw=sen_ids, experiment=experiment, num_instances=num_instances, reps_to_use=reps_to_use)

    labels = np.array(labels)

    # label_sort_inds = np.argsort(sen_ids) #TODO: check that sen_ids are ordered consistently with sentence list
    # # More robust thing to do is to use strings and a dictionary and not have an unnecessary mapping step
    # # Explore alternative querying of hippocampus - try querying with sentence id and see what stimulus is
    # labels = labels[label_sort_inds]
    # data = data[label_sort_inds, :, :]

    if noMag:
        inds_to_remove = range(2, data.shape[2], 3)
    else:
        inds_to_remove = []
    if sorted_inds is None:
        ordered_inds = range(data.shape[2])
    else:
        ordered_inds = sorted_inds

    final_inds = [i for i in ordered_inds if i not in inds_to_remove]
    data = data[:, final_inds, :]


    return data, labels, time, final_inds