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

SENSOR_MAP = '/home/nrafidi/sensormap.mat'
SAVE_RDM = '/share/volume0/nrafidi/RDM_{exp}_{tmin}_{tmax}_{word}.npz'
SAVE_SCORES = '/share/volume0/nrafidi/Scores_{exp}_{metric}_{reg}_{tmin}_{tmax}_{word}_semantics_rnng_lstm_corr.npz'
SEMANTIC_RDM = '/share/volume1/sjat/rdm/krns2-sent-disimilarity-{vsm}.pkl'
VECTORS = '/share/volume0/RNNG/sentence_stimuli_tokenized_tagged_pred_trees_no_preterms_vectors.txt'
LSTM = '/share/volume0/RNNG/test_sents_vectors_lstm.txt'

ANIMATE = ['dog', 'doctor', 'student', 'monkey']
INANIMATE = ['door', 'hammer', 'peach', 'school']
REGIONS_TO_PLOT = ['R_Frontal', 'R_Occipital', 'L_Temporal']

EXP_INDS = {'krns2': range(64, 96),
            'PassAct2': range(32, 64),
            'PassAct3': range(32)}


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


def word_len_rdm(words):
    print(words)
    num_words = words.size
    rdm = np.zeros((num_words, num_words))
    for i in range(num_words):
        for j in range(num_words):
            rdm[i, j] = np.abs(len(words[i]) - len(words[j]))
    return rdm

def word_id_rdm(words):
    print(words)
    num_words = words.size
    rdm = np.zeros((num_words, num_words))
    for i in range(num_words):
        for j in range(num_words):
            if words[i] == words[j]:
                rdm[i, j] = 0
            else:
                rdm[i, j] = 1
    return rdm


def ani_rdm(words):
    print(words)
    num_words = words.size
    rdm = np.zeros((num_words, num_words))
    for i in range(num_words):
        wordi = words[i]
        if '.' in wordi:
            wordi = wordi[:-1]
        for j in range(num_words):
            wordj = words[j]
            if '.' in wordj:
                wordj = wordj[:-1]
            if wordi in ANIMATE and wordj in ANIMATE:
                rdm[i, j] = 0
            elif wordi in INANIMATE and wordj in INANIMATE:
                rdm[i, j] = 0
            else:
                rdm[i, j] = 1
    return rdm


def rank_correlate_rdms(rdm1, rdm2):
    diagonal_offset = -1 # exclude the main diagonal
    lower_tri_inds = np.tril_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[lower_tri_inds],rdm2[lower_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


def load_sentence_data(subject, word, sen_type, experiment, proc, num_instances, reps_to_use,
                       tmin, tmax, sorted_inds=None):
    evokeds, labels, time, sen_ids = load_data.load_raw(subject, word, sen_type,
                                                        experiment=experiment, proc=proc,
                                                        tmin=tmin, tmax=tmax)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--word', default='secondNoun')
    parser.add_argument('--dist', default='euclidean')
    parser.add_argument('--tmin', type=float, default=-2.0)
    parser.add_argument('--tmax', type=float, default=1.5)
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    args = parser.parse_args()

    word = args.word
    sorted_inds, sorted_reg = sort_sensors()

    fname = SAVE_RDM.format(exp=args.experiment, tmin=args.tmin, tmax=args.tmax, word=args.word)

    vectors = np.loadtxt(VECTORS)
    vectors = vectors[EXP_INDS[args.experiment], :]

    vec_rdm = squareform(pdist(vectors, metric=args.dist))

    lstm = np.loadtxt(LSTM)
    lstm = lstm[EXP_INDS[args.experiment], :]

    lstm_rdm = squareform(pdist(lstm, metric=args.dist))

    glove_rdm_list = pickle.load(open(SEMANTIC_RDM.format(vsm='glove')))
    glove_rdm = glove_rdm_list[1]
    glove_rdm = glove_rdm[EXP_INDS[args.experiment], :]
    glove_rdm = glove_rdm[:, EXP_INDS[args.experiment]]
    # fig, ax = plt.subplots()
    # ax.imshow(glove_rdm, interpolation='nearest')

    w2v_rdm_list = pickle.load(open(SEMANTIC_RDM.format(vsm='w2v')))
    w2v_rdm = w2v_rdm_list[1]
    w2v_rdm = w2v_rdm[EXP_INDS[args.experiment], :]
    w2v_rdm = w2v_rdm[:, EXP_INDS[args.experiment]]
    # fig, ax = plt.subplots()
    # ax.imshow(w2v_rdm, interpolation='nearest')
    # plt.show()

    if os.path.isfile(fname):
        result = np.load(fname)
        rdm = result['rdm']
    else:
        rdm_by_sub_list = []
        for subject in load_data.VALID_SUBS[args.experiment]:


            act_data, labels_act, time_act = load_sentence_data(subject, args.word, 'active', args.experiment, args.proc,
                                                                args.num_instances, args.reps_to_use, args.tmin, args.tmax,
                                                                sorted_inds=sorted_inds)

            pass_data, labels_pass, time_pass = load_sentence_data(subject, args.word, 'passive', args.experiment, args.proc,
                                                                   args.num_instances, args.reps_to_use, args.tmin, args.tmax,
                                                                   sorted_inds=sorted_inds)

            min_time = np.min([time_act.size, time_pass.size])
            act_data = act_data[:, :, :min_time]
            pass_data = pass_data[:, :, :min_time]

            total_data = np.concatenate((act_data, pass_data), axis=0)
            total_labels = np.concatenate((labels_act, labels_pass), axis=0)

            rdm_by_reg_list = []
            for reg in set(sorted_reg):
                rdm_by_time_list = []
                for t in range(0, total_data.shape[2]):
                    locs = [i for i, x in enumerate(sorted_reg) if x == reg]
                    reshaped_data = np.squeeze(total_data[:, locs, t])
                    rdm = squareform(pdist(reshaped_data, metric=args.dist))
                    rdm_by_time_list.append(rdm[None, :, :])
                time_rdm = np.concatenate(rdm_by_time_list)
                print(time_rdm.shape)
                rdm_by_reg_list.append(time_rdm[None, ...])
            reg_rdm = np.concatenate(rdm_by_reg_list)
            print(reg_rdm.shape)
            rdm_by_sub_list.append(reg_rdm[None, ...])
        rdm = np.concatenate(rdm_by_sub_list)
        print(rdm.shape)

        np.savez_compressed(fname, rdm=rdm)

    rdm = np.squeeze(np.mean(rdm, axis=0))
    ap_list, sen_list = rnng_rdm.get_sen_lists()

    ap_rdm = rnng_rdm.syn_rdm(ap_list)
    ap_rdm = ap_rdm[EXP_INDS[args.experiment], :]
    ap_rdm = ap_rdm[:, EXP_INDS[args.experiment]]
    print(ap_rdm.shape)
    print(matrix_rank(ap_rdm))
    semantic_rdm = rnng_rdm.sem_rdm(sen_list, ap_list)
    semantic_rdm = semantic_rdm[EXP_INDS[args.experiment], :]
    semantic_rdm = semantic_rdm[:, EXP_INDS[args.experiment]]
    print(semantic_rdm.shape)
    print(matrix_rank(semantic_rdm))

    uni_reg = np.unique(sorted_reg)
    num_reg = rdm.shape[0]



    fig, axs = plt.subplots(len(REGIONS_TO_PLOT), 1, figsize=(20, 20))
    fig_zoom, axs_zoom = plt.subplots(len(REGIONS_TO_PLOT), 1, figsize=(20, 20))
    time = np.arange(args.tmin, args.tmax+0.002, 0.002)
    time_zoom = np.arange(0.0, args.tmax + 0.002, 0.002)
    min_reg = np.empty((num_reg,))
    max_reg = np.empty((num_reg,))
    min_reg_zoom = np.empty((num_reg,))
    max_reg_zoom = np.empty((num_reg,))
    colors = ['b', 'g', 'r', 'c']
    for i_reg, reg in enumerate(REGIONS_TO_PLOT):
        j_reg = np.where(uni_reg ==reg)
        print(uni_reg[j_reg])
        ax = axs[i_reg]
        ax_zoom = axs_zoom[i_reg]
        fname = SAVE_SCORES.format(exp=args.experiment, metric=args.dist, reg=uni_reg[j_reg], tmin=args.tmin, tmax=args.tmax, word=args.word)
        if os.path.isfile(fname):
            result = np.load(fname)
            syn_scores = result['syn_scores']
            sem_scores = result['sem_scores']
            glove_scores = result['glove_scores']
            w2v_scores = result['w2v_scores']
            rnng_scores = result['rnng_scores']
            lstm_scores = result['lstm_scores']
        else:
            syn_scores = np.empty((rdm.shape[1],))
            sem_scores = np.empty((rdm.shape[1],))
            glove_scores = np.empty((rdm.shape[1],))
            w2v_scores = np.empty((rdm.shape[1],))
            rnng_scores = np.empty((rdm.shape[1],))
            lstm_scores = np.empty((rdm.shape[1],))
            for i_t in range(rdm.shape[1]):
                syn_scores[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[j_reg, i_t, :, :]), ap_rdm)
                sem_scores[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[j_reg, i_t, :, :]), semantic_rdm)
                glove_scores[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[j_reg, i_t, :, :]), glove_rdm)
                w2v_scores[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[j_reg, i_t, :, :]), w2v_rdm)
                rnng_scores[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[j_reg, i_t, :, :]), vec_rdm)
                lstm_scores[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[j_reg, i_t, :, :]), lstm_rdm)
            np.savez_compressed(fname, syn_scores=syn_scores, sem_scores=sem_scores, glove_scores=glove_scores,
                                w2v_scores=w2v_scores, rnng_scores=rnng_scores, lstm_scores=lstm_scores)

        min_reg[i_reg] = np.min([np.min(syn_scores), np.min(glove_scores), np.min(rnng_scores), np.min(lstm_scores)])
        max_reg[i_reg] = np.max([np.max(syn_scores), np.max(glove_scores), np.max(rnng_scores), np.max(lstm_scores)])

        all_scores = np.concatenate([syn_scores[None, ...], glove_scores[None, ...], rnng_scores[None, ...], lstm_scores[None, ...], ])
        print(all_scores.shape)

        good_scores = all_scores >= 0.15
        print(good_scores.shape)

        win_scores = np.argmax(all_scores, axis=0)
        print(win_scores.shape)


        h1 = ax.plot(time, syn_scores)
        # h2 = ax.plot(time, sem_scores)
        h3 = ax.plot(time, glove_scores)
        # h4 = ax.plot(time, w2v_scores)
        h5 = ax.plot(time, rnng_scores)
        h6 = ax.plot(time, lstm_scores)
        h1[0].set_label('Syntax')
        # h2[0].set_label('Simple Semantics')
        h3[0].set_label('glove Semantics')
        # h4[0].set_label('w2v Semantics')
        h5[0].set_label('RNNG')
        h6[0].set_label('LSTM')
        ax.legend()

        # for i_time in range(all_scores.shape[-1]):
        #     if good_scores[win_scores[i_time], i_time]:
        #         ax.scatter(time[i_time], all_scores[win_scores[i_time], i_time]+0.05, c=colors[win_scores[i_time]], linewidths=0.0)
        ax.set_title(uni_reg[i_reg])
        ax.set_xlim(args.tmin, args.tmax+0.5)
        ax.set_xticks(np.arange(args.tmin, args.tmax, 0.5))

        syn_scores_zoom = syn_scores[time >= 0.0]
        glove_scores_zoom = glove_scores[time >= 0.0]
        rnng_scores_zoom = rnng_scores[time >= 0.0]
        lstm_scores_zoom = lstm_scores[time >= 0.0]
        min_reg_zoom[i_reg] = np.min(
            [np.min(syn_scores_zoom), np.min(glove_scores_zoom), np.min(rnng_scores_zoom), np.min(lstm_scores_zoom)])
        max_reg_zoom[i_reg] = np.max(
            [np.max(syn_scores_zoom), np.max(glove_scores_zoom), np.max(rnng_scores_zoom), np.max(lstm_scores_zoom)])

        all_scores_zoom = np.concatenate(
            [syn_scores_zoom[None, ...], glove_scores_zoom[None, ...], rnng_scores_zoom[None, ...], lstm_scores_zoom[None, ...], ])

        good_scores_zoom = all_scores_zoom>= 0.15

        win_scores_zoom = np.argmax(all_scores_zoom, axis=0)

        h1 = ax_zoom.plot(time_zoom, syn_scores_zoom)
        # h2 = ax.plot(time, sem_scores)
        h3 = ax_zoom.plot(time_zoom, glove_scores_zoom)
        # h4 = ax.plot(time, w2v_scores)
        h5 = ax_zoom.plot(time_zoom, rnng_scores_zoom)
        h6 = ax_zoom.plot(time_zoom, lstm_scores_zoom)
        h1[0].set_label('Syntax')
        # h2[0].set_label('Simple Semantics')
        h3[0].set_label('glove Semantics')
        # h4[0].set_label('w2v Semantics')
        h5[0].set_label('RNNG')
        h6[0].set_label('LSTM')
        ax_zoom.legend()

        for i_time in range(all_scores_zoom.shape[-1]):
            if good_scores_zoom[win_scores_zoom[i_time], i_time]:
                ax_zoom.scatter(time_zoom[i_time], all_scores_zoom[win_scores_zoom[i_time], i_time]+0.05, c=colors[win_scores_zoom[i_time]], linewidths=0.0)

        ax_zoom.set_title(uni_reg[i_reg])
        ax_zoom.set_xlim(0.0, args.tmax + 0.5)
        ax_zoom.set_xticks(np.arange(0.0, args.tmax, 0.5))

        # ax.legend([h1, h2], ['Syntax', 'Semantics'])
    max_val = np.max(max_reg)
    min_val = np.min(min_reg)
    for i_reg in range(len(REGIONS_TO_PLOT)):
        axs[i_reg].set_ylim(min_val, max_val+0.1)
    max_val_zoom = np.max(max_reg_zoom)
    min_val_zoom = np.min(min_reg_zoom)
    for i_reg in range(len(REGIONS_TO_PLOT)):
        axs_zoom[i_reg].set_ylim(min_val_zoom, max_val_zoom+0.1)

    fig.suptitle('{} {}'.format(args.experiment, args.dist))
    fig.tight_layout()
    fig_zoom.suptitle('{} {}'.format(args.experiment, args.dist))
    fig_zoom.tight_layout()
    fig.savefig('RDM_scores_{exp}_{metric}_{tmin}_{tmax}_{word}_subset.pdf'.format(exp=args.experiment, metric=args.dist, tmin=args.tmin, tmax=args.tmax, word=args.word))
    fig_zoom.savefig('RDM_scores_{exp}_{metric}_0_{tmax}_{word}_subset.pdf'.format(exp=args.experiment, metric=args.dist,
                                                                                 tmax=args.tmax,
                                                                                 word=args.word))
    # plt.savefig()
    plt.show()


    #
    #
    #
    #
    #
    #
    #
    #     best_rdm_len = np.argmax(score_rdm_len)
    #     fig, ax = plt.subplots()
    #     h = ax.imshow(rdm_list[best_rdm_len], interpolation='nearest', aspect='auto')
    #     plt.colorbar(h)
    #     ax.set_title('{} {} {} {} len'.format(reg, time_act[best_rdm_len],
    #                                    score_rdm_len[best_rdm_len], word))
    #     plt.savefig('RDM_len_{}_{}_{}.pdf'.format(reg, best_rdm_len, word))
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(time_act, score_rdm_len)
    #     ax.set_title('{} {} len'.format(reg, word))
    #     ax.set_ylim(0, 0.5)
    #     plt.savefig('Score_len_{}_{}.pdf'.format(reg, word))
    #
    #     best_rdm_id = np.argmax(score_rdm_id)
    #     fig, ax = plt.subplots()
    #     h = ax.imshow(rdm_list[best_rdm_id], interpolation='nearest', aspect='auto')
    #     plt.colorbar(h)
    #     ax.set_title('{} {} {} {} id'.format(reg, time_act[best_rdm_id],
    #                                    score_rdm_id[best_rdm_id], word))
    #     plt.savefig('RDM_id_{}_{}_{}.pdf'.format(reg, best_rdm_id, word))
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(time_act, score_rdm_id)
    #     ax.set_ylim(0, 0.5)
    #     ax.set_title('{} {} id'.format(reg, word))
    #     plt.savefig('Score_id_{}_{}.pdf'.format(reg, word))
    #
    #     best_rdm_ani = np.argmax(score_rdm_ani)
    #     fig, ax = plt.subplots()
    #     h = ax.imshow(rdm_list[best_rdm_ani], interpolation='nearest', aspect='auto')
    #     plt.colorbar(h)
    #     ax.set_title('{} {} {} {} ani'.format(reg, time_act[best_rdm_ani],
    #                                          score_rdm_id[best_rdm_ani], word))
    #     plt.savefig('RDM_ani_{}_{}_{}.pdf'.format(reg, best_rdm_ani, word))
    #
    # fig, ax = plt.subplots()
    # ax.plot(time_act, score_rdm_ani)
    # ax.set_ylim(0, 0.5)
    # ax.set_title('{} {} ani'.format(reg, word))
    # plt.savefig('Score_ani_{}_{}.pdf'.format(reg, word))
    # word_rdm_len = word_len_rdm(total_labels)
    # fig, ax = plt.subplots()
    # h = ax.imshow(word_rdm_len, interpolation='nearest')
    # ax.set_title('Word len RDM')
    # plt.colorbar(h)
    # plt.savefig('RDM_word_len_{}.pdf'.format(word))
    #
    # word_rdm_id = word_id_rdm(total_labels)
    # fig, ax = plt.subplots()
    # h = ax.imshow(word_rdm_id, interpolation='nearest')
    # ax.set_title('Word id RDM')
    # plt.colorbar(h)
    # plt.savefig('RDM_word_id_{}.pdf'.format(word))
    #
    # word_rdm_ani = ani_rdm(total_labels)
    # fig, ax = plt.subplots()
    # h = ax.imshow(word_rdm_ani, interpolation='nearest')
    # ax.set_title('Word ani RDM')
    # plt.colorbar(h)
    # plt.savefig('RDM_word_ani_{}.pdf'.format(word))
    # # plt.show()