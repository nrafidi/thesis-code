import argparse
import load_data
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
import rnng_rdm
import os.path
import pickle
import string
import Mantel

SENSOR_MAP = '/home/nrafidi/sensormap.mat'
SENTENCES = '/share/volume0/RNNG/sentence_stimuli_tokenized_tagged_with_unk_final.txt'

SAVE_MEG_RDM = '/share/volume0/RNNG/meg_rdm/RDM_{exp}_{word}_{reg}_win{win_size}_avg{avg_time}_{dist}_num{num_instances}_reps{reps_to_use}_proc{proc}.npz'
SAVE_RDM_SCORES = '/share/volume0/RNNG/results/Scores_{exp}_{metric}_{reg}_{mode}_{model}_{word}_noUNK{noUNK}_win{win_size}_avg{avg_time}_{dist}_num{num_instances}_reps{reps_to_use}_proc{proc}.npz'

HUMAN_WORDNET_SEN_RDM = '/share/volume0/RNNG/semantic_models/wordnet/sentence_similarity/{experiment}_{model}_semantic_dissimilarity.npz'
WORDNET_WORD_RDM = '/share/volume0/RNNG/semantic_models/{experiment}_all_{word}_RDM_wordnet.npz'
SEMANTIC_VECTORS = '/share/volume0/RNNG/semantic_models/nouns_verb.pkl'
RNNG_VECTORS = '/share/volume0/RNNG/sentence_stimuli_tokenized_tagged_pred_trees_no_preterms_vectors.txt'
LSTM_VECTORS = '/share/volume0/RNNG/test_sents_vectors_lstm.txt'

EXP_INDS = {'krns2': range(64, 96),
            'PassAct2': range(32, 64),
            'PassAct3': range(32)}
            # 'krns3': range(96, 156),
            # 'krns4': range(156, 216),
            # 'krns5': range(216, 336)}

VALID_MODELS = {'noun1': ['glove', 'w2v', 'wordnet'],
                'verb': ['glove', 'w2v', 'wordnet'],
                'noun2': ['glove', 'w2v', 'wordnet'],
                'sentence': ['glove-avg', 'glove-cat', 'w2v-avg', 'w2v-cat',
                             'wordnet', 'human-sem', 'human-syn', 'RNNG', 'LSTM']}

NUMAP = 96
WAS = 'was'
BY = 'by'


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


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


def get_sen_lists():
    ap_list = []
    sen_list = []
    with open(SENTENCES) as f:
        i_line = 0
        for line in f:
            if i_line >= NUMAP:
                break
            i_line += 1
            sen_list.append(string.split(line))
            if WAS in line:
                if BY in line:
                    ap_list.append('P')
                else:
                    ap_list.append('PS')
            else:
                if len(string.split(line)) == 4:
                    ap_list.append('AS')
                else:
                    ap_list.append('A')
    return ap_list, sen_list


def syn_rdm(experiment):
    ap_list, _ = get_sen_lists()
    ap_rdm = np.empty((NUMAP, NUMAP))
    for i, i_sen in enumerate(ap_list):
        for j, j_sen in enumerate(ap_list):
            if j >= NUMAP:
                break
            if i_sen == j_sen:
                ap_rdm[i, j] = 0.0
            elif i_sen in j_sen or j_sen in i_sen:
                ap_rdm[i, j] = 0.5
            else:
                ap_rdm[i, j] = 1.0
    ap_rdm = ap_rdm[EXP_INDS[experiment], :]
    ap_rdm = ap_rdm[:, EXP_INDS[experiment]]
    return ap_rdm


def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    lower_tri_inds = np.tril_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[lower_tri_inds],rdm2[lower_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


def bhy_multiple_comparisons_procedure(uncorrected_pvalues, alpha=0.05):
    # originally from Mariya Toneva
    if len(uncorrected_pvalues.shape) == 1:
        uncorrected_pvalues = np.reshape(uncorrected_pvalues, (1, -1))

    # get ranks of all p-values in ascending order
    sorting_inds = np.argsort(uncorrected_pvalues, axis=1)
    ranks = sorting_inds + 1  # add 1 to make the ranks start at 1 instead of 0

    # calculate critical values under arbitrary dependence
    dependency_constant = np.sum(1 / ranks)
    critical_values = ranks * alpha / (uncorrected_pvalues.shape[1] * dependency_constant)

    # find largest pvalue that is <= than its critical value
    sorted_pvalues = np.empty(uncorrected_pvalues.shape)
    sorted_critical_values = np.empty(critical_values.shape)
    for i in range(uncorrected_pvalues.shape[0]):
        sorted_pvalues[i, :] = uncorrected_pvalues[i, sorting_inds[i, :]]
        sorted_critical_values[i, :] = critical_values[i, sorting_inds[i, :]]
    bh_thresh = -1.0*np.ones((sorted_pvalues.shape[0],))
    for j in range(sorted_pvalues.shape[0]):
        for i in range(sorted_pvalues.shape[1] - 1, -1, -1):  # start from the back
            if sorted_pvalues[j, i] <= sorted_critical_values[j, i]:
                if bh_thresh[j] < 0:
                    bh_thresh[j] = sorted_pvalues[j, i]
                    print('threshold for row ', j, ' is:', bh_thresh[j], 'critical value:', sorted_critical_values[j, i], i)

    return bh_thresh


def load_sentence_data(subject, word, sen_type, experiment, proc, num_instances, reps_to_use, sorted_inds=None):
    evokeds, labels, time, sen_ids = load_data.load_raw(subject, word, sen_type,
                                                        experiment=experiment, proc=proc)
    data, labels, sen_ids = load_data.avg_data(evokeds, labels, sentence_ids_raw=sen_ids,
                                                           experiment=experiment,
                                                           num_instances=num_instances,
                                                           reps_to_use=reps_to_use)
    labels = np.array(labels)
    label_sort_inds = np.argsort(sen_ids)
    labels = labels[label_sort_inds]
    data = data[label_sort_inds, :, :]

    if sorted_inds is not None:
        data = data[:, sorted_inds, :]

    return data, labels, time


def load_model_rdm(experiment, word, mode, model, dist, noUNK):
    if model == 'RNNG':
        vectors = np.loadtxt(RNNG_VECTORS)
        vectors = vectors[EXP_INDS[experiment], :]
        model_rdm = squareform(pdist(vectors, metric=dist))
    elif model == 'LSTM':
        lstm = np.loadtxt(LSTM_VECTORS)
        lstm = lstm[EXP_INDS[experiment], :]
        model_rdm = squareform(pdist(lstm, metric=dist))
    elif model == 'human-syn':
        model_rdm = syn_rdm(experiment)
    elif model == 'human-sem':
        result = np.load(HUMAN_WORDNET_SEN_RDM.format(experiment=experiment.lower(), model='human'))
        result_dict = result.item()
        model_rdm = result_dict[u'dissimilarity']
    elif model == 'glove-cat' or model == 'w2v-cat':
        str_len = len(model)
        vector_dict = pickle.load(open(SEMANTIC_VECTORS))
        vectors_by_word = []
        for w in ['noun1', 'verb', 'noun2']:
            key_str = '{word}_emb_{model}'.format(word=w, model=model[:(str_len-4)])
            vectors = np.stack(vector_dict[key_str])
            vectors_by_word.append(vectors[EXP_INDS[experiment], :])
        vectors = np.concatenate(vectors_by_word, axis=1)
        model_rdm = squareform(pdist(vectors, metric=dist))
    elif model == 'glove-avg' or model == 'w2v-avg':
        str_len = len(model)
        vector_dict = pickle.load(open(SEMANTIC_VECTORS))
        vectors_by_word = []
        for w in ['noun1', 'verb', 'noun2']:
            key_str = '{word}_emb_{model}'.format(word=w, model=model[:(str_len-4)])
            vectors = np.stack(vector_dict[key_str])
            vectors_by_word.append(vectors[EXP_INDS[experiment], :])
        vectors = np.mean(np.stack(vectors_by_word), axis=0)
        model_rdm = squareform(pdist(vectors, metric=dist))
    elif model == 'glove' or model == 'w2v':
        key_str = '{word}_emb_{model}'.format(word=word, model=model)
        vector_dict = pickle.load(open(SEMANTIC_VECTORS))
        vectors = np.stack(vector_dict[key_str])
        vectors = vectors[EXP_INDS[experiment], :]
        model_rdm = squareform(pdist(vectors, metric=dist))
    else:
        assert model == 'wordnet'
        if mode == 'sentence':
            result = np.load(HUMAN_WORDNET_SEN_RDM.format(experiment=experiment.lower(), model='wordnet'))
            result_dict = result.item()
            model_rdm = result_dict[u'dissimilarity']
        else:
            model_rdm = np.empty(len(EXP_INDS[experiment]), len(EXP_INDS[experiment]))
    if noUNK:
        _, sen_list = get_sen_lists()
        good_inds = [i for i, sen in enumerate(sen_list) if 'UNK' not in sen and i in EXP_INDS['experiment']]
        model_rdm = model_rdm[good_inds, :]
        model_rdm = model_rdm[:, good_inds]
    return model_rdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--model')
    parser.add_argument('--mode')
    parser.add_argument('--word')
    parser.add_argument('--score')
    parser.add_argument('--region')
    parser.add_argument('--noUNK', default='False')
    parser.add_argument('--win_size', type=int, default=1)
    parser.add_argument('--avg_time', default='False')
    parser.add_argument('--dist', default='euclidean')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--force', default='False')
    args = parser.parse_args()

    experiment = args.experiment
    model = args.model
    mode = args.mode
    word = args.word
    score = args.score
    region = args.region
    noUNK = str_to_bool(args.noUNK)
    win_size = args.win_size
    avg_time = str_to_bool(args.avg_time)
    dist = args.dist
    num_instances = args.num_instances
    reps_to_use = args.reps_to_use
    proc = args.proc
    force = str_to_bool(args.force)

    if proc != load_data.DEFAULT_PROC:
        proc_str = proc
    else:
        proc_str = 'default'

    results_fname = SAVE_RDM_SCORES.format(exp=experiment, metric=score, reg=region, mode=mode, model=model, word=word,
                                           noUnk=bool_to_str(noUNK), win_size=win_size, avg_time=bool_to_str(avg_time),
                                           dist=dist, num_instances=num_instances, reps_to_use=reps_to_use, proc=proc_str)

    if model not in VALID_MODELS[mode]:
        print('Model {} not available for mode {}.'.format(model, mode))
    elif avg_time and win_size == 1:
        print('No point in averaging a single timepoint.')
    elif os.path.isfile(results_fname) and not force:
        print('Results file already exists: {}'.format(results_fname))
    else:
        # Get brain regions from sensors
        sorted_inds, sorted_reg = sort_sensors()

        #MEG Data RDM
        meg_fname = SAVE_MEG_RDM.format(exp=experiment, word=word, reg=region, win_size=win_size,
                                        avg_time=bool_to_str(avg_time), dist=dist, num_instances=num_instances,
                                        reps=reps_to_use, proc=proc_str)

        if os.path.isfile(meg_fname):
            result = np.load(meg_fname)
            brain_rdm = result['rdm']
        else:
            rdm_by_sub_list = []
            for subject in load_data.VALID_SUBS[experiment]:

                act_data, labels_act, time_act = load_sentence_data(subject, word, 'active', experiment, proc,
                                                                    num_instances, reps_to_use,
                                                                    sorted_inds=sorted_inds)

                pass_data, labels_pass, time_pass = load_sentence_data(subject, word, 'passive', experiment,
                                                                       proc, num_instances, reps_to_use,
                                                                       sorted_inds=sorted_inds)

                min_time = np.min([time_act.size, time_pass.size])
                act_data = act_data[:, :, :min_time]
                pass_data = pass_data[:, :, :min_time]

                total_data = np.concatenate((act_data, pass_data), axis=0)
                total_labels = np.concatenate((labels_act, labels_pass), axis=0)

                rdm_by_time_list = []
                for t in range(0, total_data.shape[2]):
                    locs = [i for i, x in enumerate(sorted_reg) if x == region]
                    reshaped_data = np.squeeze(total_data[:, locs, t])
                    rdm = squareform(pdist(reshaped_data, metric=dist))
                    rdm_by_time_list.append(rdm[None, :, :])
                time_rdm = np.concatenate(rdm_by_time_list)
                print(time_rdm.shape)
                rdm_by_sub_list.append(time_rdm[None, ...])
            brain_rdm = np.concatenate(rdm_by_sub_list)
            print(brain_rdm.shape)

            np.savez_compressed(meg_fname, rdm=brain_rdm, labels=total_labels)

        model_rdm = load_model_rdm(experiment, word, mode, model, dist, noUNK)

        num_time = brain_rdm.shape[0]
        rdm_scores = np.empty(num_time)
        rdm_pvals = np.empty(num_time)
        for i_t in range(num_time):
            if score == 'kendalltau':
                rdm_scores[i_t], rdm_pvals[i_t] = ktau_rdms(np.squeeze(brain_rdm[i_t, :, :]), model_rdm)
            else:
                rdm_scores[i_t], rdm_pvals[i_t] = Mantel.test(np.squeeze(brain_rdm[i_t, :, :]), model_rdm, tail='upper')
        bh_thresh = bhy_multiple_comparisons_procedure(rdm_pvals)
        np.savez_compressed(results_fname, rdm_scores=rdm_scores, rdm_pvals=rdm_pvals, bh_thresh=bh_thresh)

