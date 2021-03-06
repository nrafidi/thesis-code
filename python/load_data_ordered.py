import numpy as np
import argparse
import hippo.io
import hippo.query
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
import string
import scipy.io as sio
import path_constants

NUM_SENTENCES = 16
NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']}

EXP_INDS = {'krns2': range(64, 96),
            'PassAct2': range(32, 64),
            'PassAct3': range(32)}

USI_NAME = {'krns2': 'krns2',
            'PassAct2': 'pass-act-2',
            'PassAct3': 'pass-act-3'}

TIME_LIMITS = {'active': {'noun1': {'tmin': -1.0, 'tmax': 4.0},
                          'verb': {'tmin': -1.5, 'tmax': 3.0},
                          'noun2': {'tmin': -2.5, 'tmax': 1.5},
                          'agent': {'tmin': -1.0, 'tmax': 1.5},
                          'patient': {'tmin': -1.0, 'tmax': 1.5}},
               'passive': {'noun1': {'tmin': -1.0, 'tmax': 4.0},
                           'verb': {'tmin': -1.5, 'tmax': 3.0},
                           'noun2': {'tmin': -2.5, 'tmax': 1.5},
                           'agent': {'tmin': -1.0, 'tmax': 1.5},
                           'patient': {'tmin': -1.0, 'tmax': 1.5}}}

noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
to_be_verbs = {'be', 'am', 'is', 'are', 'being', 'was', 'were', 'been'}
punctuation_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

IRR_WORDS = {'krns2': ['was', 'by'],
             'PassAct2': ['was', 'by', 'the'],
             'PassAct3': ['was', 'by', 'the']}

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

# Old slugs:
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'sss_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-5-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_lp-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# DEFAULT_PROC = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
DEFAULT_PROC = 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_lp-150_notch-60-120_beatremoval-first_blinkremoval-first'
PDTW_FILE = '/share/volume0/newmeg/{exp}/avg/{exp}_{sub}_{proc}_parsed_{word}_pdtwSyn.mat'
PA3_FILE = '/share/volume0/newmeg/PassAct3/parsed/{sub}/hippoParse_parsed.mat'


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


def get_sen_num_from_id(sen_id):
    m = re.match('.*sentence-(\d+)', sen_id)
    return int(m.group(1))


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
        # for i_word, word in enumerate(words):
        #     if 'punched' in word:
        #         words[i_word] = 'unklced'
        corr_words = np.array([punctuation_regex.sub('', word).lower() for word in words])
        sorted_inds = np.argsort(word_indices)
        recon = ' '.join(corr_words[sorted_inds])
        recon_sentences.append(recon)
        sentence_id_by_recon.append(sentence_id)
    return recon_sentences, sentence_id_by_recon


def order_sentences(usis, experiment):
    recon_sentences, sentence_id_by_recon = recon_sen_from_usis(usis)
    with open(path_constants.SENTENCES) as f:
        loaded_sentences = f.readlines()
    loaded_sentences = [sen.strip() for sen in loaded_sentences]
    exp_sentences = [punctuation_regex.sub('', loaded_sentences[ind]).lower().strip().replace(' ', '').replace('\t', ' ') for ind in EXP_INDS[experiment]]
    sorted_inds = [recon_sentences.index(sen) for sen in exp_sentences if sen in recon_sentences]
    sorted_sentence_ids = [sentence_id_by_recon[ind] for ind in sorted_inds]
    test_sort = [recon_sentences[ind] for ind in sorted_inds]
    assert test_sort == exp_sentences
    return sorted_sentence_ids, exp_sentences


def load_raw(subject, experiment, filters, tmin, tmax, proc=DEFAULT_PROC):
    # for example:
    # evoked, labels, sentence_ids, time = load_raw('A', 'PassAct3', [is_in_active, is_first_noun], tmin=-0.7, tmax=2.5)

    usis = hippo.query.query_usis([('stimuli_set', USI_NAME[experiment])],
                                  include_annotations=[
                                      'stimulus',
                                      'sentence_id',
                                      'word_index_in_sentence',
                                      'stanford_2017_06_09_pos',
                                      'question_id'])

    # sort in text file sentence order
    sorted_sentence_ids, sorted_sentences = order_sentences(usis, experiment)
    print sorted_sentence_ids
    # group by sentence ids
    sentence_id_to_usis = dict()
    for usi in usis:
        annotations = usis[usi]
        # filter out question words on this side, since None is not handled correctly by hippo
        if annotations['question_id'] is not None:
            continue
        if annotations['stanford_2017_06_09_pos'] is None:
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
    labels = []
    for usi in usis:
        usi_idx = sorted_sentence_ids.index(usis[usi]['sentence_id'])
        words_to_add = sorted_sentences[usi_idx].split()
        label_to_add = [w for w in words_to_add if w not in IRR_WORDS[experiment]]
        labels.append(label_to_add)

    exp_sub = [(experiment, subject)]
    uels = hippo.query.get_uels_from_usis(usis.keys(), experiment_subjects=exp_sub)
    id_uels = [(k, uels[k]) for k in uels.keys()]  # putting uels in a list instead of a map (explicit ordering)

    sentence_ids = [usis[k]['sentence_id'] for k, _ in id_uels]
    print(sentence_ids)

    sorted_inds_sentence = [sentence_ids.index(sen_id) for sen_id in sorted_sentence_ids if sen_id in sentence_ids]
    print(sorted_inds_sentence)
    labels = [labels[ind] for ind in sorted_inds_sentence]
    sentence_ids = [sentence_ids[ind] for ind in sorted_inds_sentence]

    print(labels)
    print(sentence_ids)
    assert len(labels) == len(sentence_ids)

    _, uels = zip(*id_uels)

    evokeds = np.array(
        [hippo.io.load_mne_epochs(us, preprocessing=proc, baseline=None, tmin=tmin, tmax=tmax) for us in uels])

    evokeds = evokeds[np.array(sorted_inds_sentence), ...]
    # Downsample
    evokeds = evokeds[:, :, :, ::2]
    time = np.arange(tmin, tmax + 2e-3, 2e-3)
    if evokeds.shape[3] != time.size:
        min_size = np.min([evokeds.shape[3], time.size])
        evokeds = evokeds[:, :, :, :min_size]
        time = time[:min_size]

    return evokeds, labels, time, sentence_ids


def load_PassAct3_matlab(subject, sen_type, num_instances, reps_to_use, noMag=False, sorted_inds=None):
    pa3_file = PA3_FILE.format(sub=subject)
    result = sio.loadmat(pa3_file)
    if sen_type == 'active':
        inds = range(16)
    else:
        inds = range(16, 32)
    full_data = result['fullData']
    full_data = np.stack(np.squeeze(full_data[:, inds]))
    time = result['fullTime']
    labels = np.squeeze(result['words']).tolist()
    labels = [np.squeeze(w) for w in labels]
    labels = [[str(w[0]) for w in wo if w not in IRR_WORDS['PassAct3']] for wo in labels]
    labels = [l for i_l, l in enumerate(labels) if i_l in inds]
    data, labels, sen_ids = avg_data(
        full_data, labels,
        sentence_ids_raw=inds, experiment='PassAct3', num_instances=num_instances, reps_to_use=reps_to_use)
    if noMag:
        inds_to_remove = range(2, data.shape[1], 3)
    else:
        inds_to_remove = []
    if sorted_inds is None:
        ordered_inds = range(data.shape[1])
    else:
        ordered_inds = sorted_inds

    final_inds = [i for i in ordered_inds if i not in inds_to_remove]
    data = data[:, final_inds, :]

    return data, labels, time, final_inds



def load_sentence_data(subject, word, sen_type, experiment, proc, num_instances, reps_to_use, noMag=False,
                       sorted_inds=None, tmin=None, tmax=None):
    # evoked, labels, sentence_ids, time = load_raw('A', 'PassAct3', [is_in_active, is_first_noun], tmin=-0.7, tmax=2.5)

    if tmin is None:
        tmin = TIME_LIMITS[sen_type][word]['tmin']
    if tmax is None:
        tmax = TIME_LIMITS[sen_type][word]['tmax']

    if sen_type == 'active':
        if word == 'noun1':
            filters = [is_in_active, is_first_noun]
        elif word == 'verb':
            filters = [is_in_active, is_first_non_to_be_verb]
        elif word == 'noun2':
            filters = [is_in_active, is_second_noun]
        elif word == 'agent':
            filters = [is_in_active, is_first_noun]
        else:
            filters = [is_in_active, is_second_noun]
    else:
        if word == 'noun1':
            filters = [is_in_passive, is_first_noun]
        elif word == 'verb':
            filters = [is_in_passive, is_first_non_to_be_verb]
        elif word == 'noun2':
            filters = [is_in_passive, is_second_noun]
        elif word == 'agent':
            filters = [is_in_passive, is_second_noun]
        else:
            filters = [is_in_passive, is_first_noun]

    evokeds, labels, time, sen_ids = load_raw(subject, experiment, filters, tmin, tmax)

    data, labels, sen_ids = avg_data(
        evokeds, labels,
        sentence_ids_raw=sen_ids, experiment=experiment, num_instances=num_instances, reps_to_use=reps_to_use)

    labels = np.array(labels)

    if noMag:
        inds_to_remove = range(2, data.shape[1], 3)
    else:
        inds_to_remove = []
    if sorted_inds is None:
        ordered_inds = range(data.shape[1])
    else:
        ordered_inds = sorted_inds

    final_inds = [i for i in ordered_inds if i not in inds_to_remove]
    data = data[:, final_inds, :]

    return data, labels, time, final_inds


def plot_data_array(data, time, sen_type):
    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    fig, ax = plt.subplots()
    h = ax.imshow(data[sorted_inds, :], interpolation='nearest', aspect='auto')
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, len(time), 250))
    label_time = time[::250]
    label_time[np.abs(label_time) < 1e-15] = 0.0
    ax.set_xticklabels(label_time)
    ax.set_xlabel('Time')
    if sen_type == 'active':
        text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
        max_line = 2.51 * 500
    else:
        text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
        max_line = 3.51 * 500

    for i_v, v in enumerate(np.arange(0.5 * 500, max_line, 0.5 * 500)):
        ax.axvline(x=v, color='k')
        if i_v < len(text_to_write):
            plt.text(v + 0.05 * 500, 15, text_to_write[i_v])

    return fig, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sen_type')
    args = parser.parse_args()
    data, labels, time, final_inds = load_sentence_data(subject='A',
                                                        word='noun1',
                                                        sen_type=args.sen_type,
                                                        experiment='PassAct3',
                                                        proc=DEFAULT_PROC,
                                                        num_instances=1,
                                                        reps_to_use=10,
                                                        noMag=False,
                                                        sorted_inds=None)

    print(data.shape)
    new_labels = [lab if len(lab) > 2 else [lab[0], lab[1], ''] for lab in labels]
    short_sens = [len(lab) == 2 for lab in labels]
    new_labels = np.array(new_labels)
    print(new_labels)
    print(np.array(short_sens))

    fig0, ax0 = plot_data_array(np.squeeze(np.mean(data, axis=0)), time, args.sen_type)
    ax0.set_title('All Data')
    fig1, ax1 = plot_data_array(np.squeeze(np.mean(data[short_sens, :, :], axis=0)), time, args.sen_type)
    ax1.set_title('Short Sentences')
    fig2, ax2 = plot_data_array(np.squeeze(np.mean(data[np.logical_not(short_sens), :, :], axis=0)), time, args.sen_type)
    ax2.set_title('Long Sentences')

    assert not np.all(np.mean(data, axis=0) == np.mean(data[np.logical_not(short_sens), :, :], axis=0))

    sen_set = np.unique(new_labels, axis=0).tolist()
    num_labels = new_labels.shape[0]
    sen_ints = np.empty((num_labels,))
    for i_l in range(num_labels):
        for j_l, l in enumerate(sen_set):
            if np.all(l == new_labels[i_l, :]):
                sen_ints[i_l] = j_l
                break
    print(sen_set)
    plt.show()