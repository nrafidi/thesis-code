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

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

ANIMATE = ['dog', 'doctor', 'student', 'monkey']
INANIMATE = ['door', 'hammer', 'peach', 'school']

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


def load_sentence_data(subject, word, sen_type, experiment, proc, num_instances, reps_to_use, sorted_inds=None):
    evokeds, labels, time, sen_ids = load_data.load_raw(subject, word, sen_type,
                                                        experiment=experiment, proc=proc,
                                                        tmin=-0.5, tmax=1.5)
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
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    args = parser.parse_args()

    word = args.word
    sorted_inds, sorted_reg = sort_sensors()

    rdm_by_sub_list = []
    for subject in ['B', 'C']: #load_data.VALID_SUBS[args.experiment]:


        act_data, labels_act, time_act = load_sentence_data(subject, args.word, 'active', args.experiment, args.proc,
                                                            args.num_instances, args.reps_to_use,
                                                            sorted_inds=sorted_inds)

        pass_data, labels_pass, time_pass = load_sentence_data(subject, args.word, 'passive', args.experiment, args.proc,
                                                               args.num_instances, args.reps_to_use,
                                                               sorted_inds=sorted_inds)

        min_time = np.min([time_act.size, time_pass.size])
        act_data = act_data[:, :, :min_time]
        pass_data = pass_data[:, :, :min_time]

        total_data = np.concatenate((act_data, pass_data), axis=0)
        total_labels = np.concatenate((labels_act, labels_pass), axis=0)

        rdm_by_reg_list = []
        for reg in set(sorted_reg):
            rdm_by_time_list = []
            score_rdm_len = np.zeros((total_data.shape[2],))
            score_rdm_id = np.zeros((total_data.shape[2],))
            score_rdm_ani = np.zeros((total_data.shape[2],))
            for t in range(0, total_data.shape[2]):
                locs = [i for i, x in enumerate(sorted_reg) if x == reg]
                reshaped_data = np.squeeze(total_data[:, locs, t])
                rdm = squareform(pdist(reshaped_data))
                rdm_by_time_list.append(rdm[None, :, :])
            time_rdm = np.concatenate(rdm_by_time_list)
            print(time_rdm.shape)
            rdm_by_reg_list.append(time_rdm[None, ...])
        reg_rdm = np.concatenate(rdm_by_reg_list)
        print(reg_rdm.shape)
        rdm_by_sub_list.append(reg_rdm[None, ...])
    rdm = np.concatenate(rdm_by_sub_list)
    print(rdm.shape)
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
    fig, axs = plt.subplots(num_reg, 1)
    for i_reg in range(num_reg):
        print(uni_reg[i_reg])
        ax = axs[i_reg]
        syn_scores = np.empty((rdm.shape[1],))
        sem_scores = np.empty((rdm.shape[1],))
        for i_t in range(rdm.shape[1]):
            syn_scores[i_t], _ = kendalltau(np.squeeze(rdm[i_reg, i_t, :, :]), ap_rdm)
            sem_scores[i_t], _ = kendalltau(np.squeeze(rdm[i_reg, i_t, :, :]), semantic_rdm)
        h1 = ax.plot(time_act, syn_scores)
        h2 = ax.plot(time_act, sem_scores)
        h1[0].set_label('Syntax')
        h2[0].set_label('Semantics')
        ax.legend()
        ax.set_title(uni_reg[i_reg])
        # ax.legend([h1, h2], ['Syntax', 'Semantics'])
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