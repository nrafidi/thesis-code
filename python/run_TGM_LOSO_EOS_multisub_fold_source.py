import argparse
from rnng_new import load_data
import models
import numpy as np
import os.path
import random
import warnings
import mne

TOP_DIR = '/share/volume0/nrafidi/PassAct3_TGM_LOSO_EOS_SOURCE/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_{reg}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'

VALID_SUBS = ['B', 'G', 'C', 'D', 'E', 'P', 'R', 'T', 'U', 'V']
VALID_ALGS = ['lr-l2', 'lr-l1']
VALID_SEN_TYPE = ['active', 'passive', 'pooled']
VALID_WORDS = ['noun1', 'noun2', 'verb', 'voice', 'agent', 'patient', 'propid', 'senlen']

VALID_REGS = ['superiorfrontal', 'caudalmiddlefrontal', 'rostralmiddlefrontal', 'parsopercularis', 'parsorbitalis',
              'parstriangularis', 'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole', 'paracentral',
              'precentral', 'insula', 'postcentral', 'inferiorparietal', 'supramarginal', 'superiorparietal',
              'precuneus', 'cuneus', 'lateraloccipital', 'lingual', 'pericalcarine', 'isthmuscingulate',
              'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate', 'entorhinal',
              'parahippocampal', 'temporalpole', 'fusiform', 'superiortemporal', 'inferiortemporal', 'middletemporal',
              'transversetemporal', 'bankssts', 'corpuscallosum']

VALID_HEMIS = ['lh', 'rh']

TMAX=2.0

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


def _get_region_data(epochs, inv_op, filtered_usi_events, num_instances,
                     indices_in_master_experiment_stimuli):
    print(epochs)
    print(filtered_usi_events)
    multi_instance_usi_events = list()
    for (usi_, events_), index_in_master in zip(filtered_usi_events, indices_in_master_experiment_stimuli):
        for i in range(num_instances):
            instance_events = [
                events_[j] for j in range(i, len(events_), num_instances)]
            if len(instance_events) == 0:
                # we raise here because downstream analysis becomes complicated if we need to remember
                # a jagged number of instances
                raise ValueError('Unable to produce meg_settings.num_output_instances_per_key instances')
            multi_instance_usi_events.append((usi_, instance_events))
    filtered_usi_events = multi_instance_usi_events
    print(filtered_usi_events)
    evoked = list()
    for _, ev in filtered_usi_events:
        ev_epochs = epochs[ev]
        # mne will silently allow bad events if there is at least one good event; this assert stops that
        assert(len(ev_epochs) == len(ev))
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            evoked.append(ev_epochs.average())
    print(evoked)
    source_estimates = list()
    # this for loop takes about 30 seconds. kinda intractable...
    for e in evoked:
        inv = mne.minimum_norm.prepare_inverse_operator(
                inv_op, e.nave, lambda2=1./9., method='dSPM', verbose=False)
        source_estimates.append(
            mne.minimum_norm.apply_inverse(e, inv, prepared=True, verbose=False))
    print(source_estimates)
    source_data = np.concatenate([source_estimate.data[None, ...] for source_estimate in source_estimates],
                                 axis=0)
    print('Source data shape: {}'.format(source_data.shape))
    return source_data


# Runs the TGM experiment
def run_tgm_exp(sen_type,
                word,
                region,
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
    top_dir = TOP_DIR
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)

    fname = SAVE_FILE.format(dir=top_dir,
                             sen_type=sen_type,
                             word=word,
                             reg=region,
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

    if sen_type == 'pooled':
        voice = ['active', 'passive']
    else:
        voice = [sen_type]

    data_list = []
    sen_ints = []
    time = []
    filter_sets = [load_data.map_alignment_to_filters(v, 'last') for v in voice]
    for i_sub, subject in enumerate(VALID_SUBS):

        inv_op, region_labels = load_data.load_inverse_operator(
            subject=subject, experiment='PassAct3',
            proc=proc, structural_label_regex=region)

        epochs, usi_events, _, sen_ints_sub, time_sub = load_data.load_epochs(
            subject=subject, experiment='PassAct3', filter_sets=filter_sets,
            tmin=0.0, tmax=TMAX)

        data = _get_region_data(epochs=epochs,
                                inv_op=inv_op,
                                filtered_usi_events=usi_events,
                                num_instances=num_instances,
                                indices_in_master_experiment_stimuli=sen_ints)

        data_list.append(data)
        if i_sub == 0:
            sen_ints = sen_ints_sub
            time = time_sub
        else:
            assert np.all(sen_ints == sen_ints_sub)
            assert np.all(time == time)

    stimuli_voice = list(load_data.read_stimuli('PassAct3'))
    # print(stimuli_voice)
    if word == 'propid':
        all_words = [stimuli_voice[sen_int]['stimulus'].split() for sen_int in sen_ints]
        all_voices = [stimuli_voice[sen_int]['voice'] for sen_int in sen_ints]
        content_words = []
        valid_inds = []
        for i_word_list, word_list in enumerate(all_words):
            curr_voice = all_voices[i_word_list]
            if len(word_list) > 5:
                valid_inds.append(i_word_list)
                content_words.append([word_list[WORD_COLS[curr_voice]['agent']], word_list[WORD_COLS[curr_voice]['verb']],
                                      word_list[WORD_COLS[curr_voice]['patient']]])
        uni_content, labels = np.unique(np.array(content_words), axis=0, return_inverse=True)
        print(np.array(content_words))
        print(uni_content)
        print(labels)
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
                if len(word_list) > 5:
                    valid_inds.append(i_sen_int)
                    labels.append(word_list[WORD_COLS[curr_voice][word]])
            else:
                labels.append(word_list[WORD_COLS[curr_voice][word]])
                valid_inds.append(i_sen_int)

    valid_inds = np.array(valid_inds)
    data_list = [data[valid_inds, ...] for data in data_list]
    sen_ints = [sen for i_sen, sen in enumerate(sen_ints) if i_sen in valid_inds]


    # print(labels)
    tmin = time.min()
    tmax = time.max()

    total_win = int((tmax - tmin) * 500)

    if win_len < 0:
        win_len = total_win - overlap

    win_starts = range(0, total_win - win_len, overlap)

    if isPerm:
        sen_ints = np.array(sen_ints)
        labels = np.array(labels)
        print(sen_ints)
        print(labels)
        uni_sen_ints, uni_inds = np.unique(sen_ints, return_index=True)
        uni_labels = labels[uni_inds]
        random.seed(random_state_perm)
        random.shuffle(uni_labels)
        for i_uni_sen, uni_sen in enumerate(uni_sen_ints):
            is_sen = sen_ints == uni_sen
            labels[is_sen] = uni_labels[i_uni_sen]
        print(sen_ints)
        print(labels)


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
    parser.add_argument('--sen_type', choices=VALID_SEN_TYPE)
    parser.add_argument('--word', choices=VALID_WORDS)
    parser.add_argument('--reg', choices=VALID_REGS)
    parser.add_argument('--hemi', choices=VALID_HEMIS)
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
    parser.add_argument('--force', default='False', choices=['True', 'False'])
    parser.add_argument('--fold', type=int)

    args = parser.parse_args()
    print(args)
    # Check that parameter setting is valid
    total_valid = True
    if args.word in ['voice', 'noun1', 'senlen', 'propid']:
        is_valid = args.sen_type == 'pooled'
        total_valid = total_valid and is_valid
        if not is_valid:
            print('{} task only valid if sen_type == pooled'.format(args.word))
    if args.fold > 15:
        is_valid = args.sen_type == 'pooled'
        total_valid = total_valid and is_valid
        if not is_valid:
            print('Incorrect number of folds')

    if total_valid:
        run_tgm_exp(sen_type=args.sen_type,
                    word=args.word,
                    region='{}-{}'.format(args.reg, args.hemi),
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
    else:
        print('Experiment parameters not valid. Skipping job.')
