import argparse
import load_data
import matplotlib
matplotlib.use('Agg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--word', default='secondNoun')
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    args = parser.parse_args()

    word = args.word
    sorted_inds, sorted_reg = sort_sensors()

    evokeds, labels, time_act, sen_ids = load_data.load_raw(args.subject, word, 'active',
                                               experiment=args.experiment, proc=args.proc, tmin=0.0, tmax=0.5)
    act_data, labels_act, sen_ids_act = load_data.avg_data(evokeds, labels, sentence_ids_raw=sen_ids, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)
    labels_act = np.array(labels_act)
    label_sort_inds = np.argsort(labels_act)
    labels_act = labels_act[label_sort_inds]
    act_data = act_data[label_sort_inds, :, :]
    act_data = act_data[:, sorted_inds, :]
    # act_data = np.squeeze(np.mean(act_data, axis=2))


    evokeds, labels, time_pass, sen_ids = load_data.load_raw(args.subject, word, 'passive',
                                               experiment=args.experiment, proc=args.proc, tmin=0.0, tmax=0.5)
    pass_data, labels_pass, sen_ids_pass = load_data.avg_data(evokeds, labels, sentence_ids_raw=sen_ids, experiment=args.experiment,
                                              num_instances=args.num_instances, reps_to_use=args.reps_to_use)
    labels_pass = np.array(labels_pass)


    min_time = np.min([time_act.size, time_pass.size])
    act_data = act_data[:, :, :min_time]
    pass_data = pass_data[:, :, :min_time]

    label_sort_inds = np.argsort(labels_pass)

    pass_data = pass_data[label_sort_inds, :, :]
    pass_data = pass_data[:, sorted_inds, :]
    # pass_data = np.squeeze(np.mean(pass_data, axis=2))
    print(act_data.shape)
    print(pass_data.shape)


    labels_pass = labels_pass[label_sort_inds]

    total_data = np.concatenate((act_data, pass_data), axis=0)
    total_labels = np.concatenate((labels_act, labels_pass), axis=0)

    word_rdm_len = word_len_rdm(total_labels)
    fig, ax = plt.subplots()
    h = ax.imshow(word_rdm_len, interpolation='nearest')
    ax.set_title('Word len RDM')
    plt.colorbar(h)
    plt.savefig('RDM_word_len.pdf')

    word_rdm_id = word_id_rdm(total_labels)
    fig, ax = plt.subplots()
    h = ax.imshow(word_rdm_id, interpolation='nearest')
    ax.set_title('Word id RDM')
    plt.colorbar(h)
    plt.savefig('RDM_word_id.pdf')


    for reg in set(sorted_reg): #['L_Occipital', 'R_Occipital']:
        rdm_list = []
        score_rdm_len = np.zeros((total_data.shape[2],))
        score_rdm_id = np.zeros((total_data.shape[2],))
        for t in range(0, total_data.shape[2]):
            locs = [i for i, x in enumerate(sorted_reg) if x == reg]
            reshaped_data = np.squeeze(total_data[:, locs, t]) #np.reshape(total_data[:, locs, :], (total_data.shape[0], -1))

            rdm = squareform(pdist(reshaped_data))
            score_rdm_len[t], _ = kendalltau(rdm, word_rdm_len)
            score_rdm_id[t], _ = kendalltau(rdm, word_rdm_id)
            rdm_list.append(rdm)

        best_rdm_len = np.argmax(score_rdm_len)
        fig, ax = plt.subplots()
        h = ax.imshow(rdm_list[best_rdm_len], interpolation='nearest')
        ax.set_title('{} {} {}'.format(reg, time_act[best_rdm_len],
                                       score_rdm_len[best_rdm_len]))
        plt.colorbar(h)
        plt.savefig('RDM_len_{}_{}.pdf'.format(reg, best_rdm_len))

        best_rdm_id = np.argmax(score_rdm_id)
        fig, ax = plt.subplots()
        h = ax.imshow(rdm_list[best_rdm_id], interpolation='nearest')
        ax.set_title('{} {} {}'.format(reg, time_act[best_rdm_id],
                                       score_rdm_id[best_rdm_id]))
        plt.colorbar(h)
        plt.savefig('RDM_id_{}_{}.pdf'.format(reg, best_rdm_id))
    # plt.show()