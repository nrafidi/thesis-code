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
import Mantel

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'
SAVE_RDM = '/share/volume0/nrafidi/RDM_{exp}_{tmin}_{tmax}_{word}.npz'
SAVE_SCORES = '/share/volume0/nrafidi/Scores_{exp}_{metric}_{reg}_{tmin}_{tmax}_{word}_rnng_lstm_mantel-upper.npz'
VECTORS = '/share/volume0/RNNG/sentence_stimuli_tokenized_tagged_pred_trees_no_preterms_vectors.txt'
LSTM = '/share/volume0/RNNG/test_sents_vectors_lstm.txt'

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    uni_reg = np.unique(sorted_reg)
    time = np.arange(args.tmin, args.tmax + 0.002, 0.002)

    kendall_scores_by_exp = []
    mantel_scores_by_exp = []
    mantel_pvals_by_exp = []
    for experiment in ['krns2', 'PassAct2']:
        fname = SAVE_RDM.format(exp=experiment, tmin=args.tmin, tmax=args.tmax, word=args.word)

        vectors = np.loadtxt(VECTORS)
        vectors = vectors[EXP_INDS[experiment], :]

        vec_rdm = squareform(pdist(vectors, metric=args.dist))

        lstm = np.loadtxt(LSTM)
        lstm = lstm[EXP_INDS[experiment], :]

        lstm_rdm = squareform(pdist(lstm, metric=args.dist))


        if os.path.isfile(fname):
            result = np.load(fname)
            rdm = result['rdm']
        else:
            rdm_by_sub_list = []
            for subject in load_data.VALID_SUBS[experiment]:


                act_data, labels_act, time_act = load_sentence_data(subject, args.word, 'active', experiment, args.proc,
                                                                    args.num_instances, args.reps_to_use, args.tmin, args.tmax,
                                                                    sorted_inds=sorted_inds)

                pass_data, labels_pass, time_pass = load_sentence_data(subject, args.word, 'passive', experiment, args.proc,
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


        num_reg = rdm.shape[0]

        fig, axs = plt.subplots(num_reg, 1, figsize=(20, 20))
        fig_mantel, axs_mantel = plt.subplots(num_reg, 1, figsize=(20, 20))


        time_zoom = np.arange(0.0, args.tmax + 0.002, 0.002)
        min_reg = np.empty((num_reg,))
        max_reg = np.empty((num_reg,))
        min_reg_mantel = np.empty((num_reg,))
        max_reg_mantel = np.empty((num_reg,))
        colors = ['b', 'g', 'r', 'c']
        kendall_scores_by_reg = []
        mantel_scores_by_reg = []
        mantel_pvals_by_reg = []
        for i_reg in range(num_reg):
            print(uni_reg[i_reg])
            ax = axs[i_reg]
            ax_mantel = axs_mantel[i_reg]
            fname = SAVE_SCORES.format(exp=experiment, metric=args.dist, reg=uni_reg[i_reg], tmin=args.tmin, tmax=args.tmax, word=args.word)
            if os.path.isfile(fname):
                result = np.load(fname)
                rnng_scores_rank = result['rnng_scores_rank']
                lstm_scores_rank = result['lstm_scores_rank']
                rnng_scores_mantel = result['rnng_scores_mantel']
                lstm_scores_mantel = result['lstm_scores_mantel']
                rnng_pvals_mantel = result['rnng_pvals_mantel']
                lstm_pvals_mantel = result['lstm_pvals_mantel']
            else:
                rnng_scores_rank = np.empty((rdm.shape[1],))
                lstm_scores_rank = np.empty((rdm.shape[1],))
                rnng_scores_mantel = np.empty((rdm.shape[1],))
                lstm_scores_mantel = np.empty((rdm.shape[1],))
                rnng_pvals_mantel = np.empty((rdm.shape[1],))
                lstm_pvals_mantel = np.empty((rdm.shape[1],))
                for i_t in range(rdm.shape[1]):
                    rnng_scores_rank[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[i_reg, i_t, :, :]), vec_rdm)
                    lstm_scores_rank[i_t], _ = rank_correlate_rdms(np.squeeze(rdm[i_reg, i_t, :, :]), lstm_rdm)
                    rnng_scores_mantel[i_t], rnng_pvals_mantel[i_t], _ = Mantel.test(np.squeeze(rdm[i_reg, i_t, :, :]), vec_rdm, tail='upper')
                    lstm_scores_mantel[i_t], lstm_pvals_mantel[i_t], _ = Mantel.test(np.squeeze(rdm[i_reg, i_t, :, :]), lstm_rdm, tail='upper')
                np.savez_compressed(fname, rnng_scores_rank=rnng_scores_rank, lstm_scores_rank=lstm_scores_rank,
                                    rnng_scores_mantel=rnng_scores_mantel, lstm_scores_mantel=lstm_scores_mantel,
                                    rnng_pvals_mantel=rnng_pvals_mantel, lstm_pvals_mantel=lstm_pvals_mantel)

            min_reg[i_reg] = np.min([np.min(rnng_scores_rank), np.min(lstm_scores_rank)])
            max_reg[i_reg] = np.max([np.max(rnng_scores_rank), np.max(lstm_scores_rank)])
            min_reg_mantel[i_reg] = np.min([np.min(rnng_scores_mantel), np.min(lstm_scores_mantel)])
            max_reg_mantel[i_reg] = np.max([np.max(rnng_scores_mantel), np.max(lstm_scores_mantel)])

            all_scores_rank = np.concatenate([rnng_scores_rank[None, ...], lstm_scores_rank[None, ...], ])
            print(all_scores_rank.shape)
            kendall_scores_by_reg.append(all_scores_rank[None, ...])
            all_pvals_mantel = np.concatenate([rnng_pvals_mantel[None, ...], lstm_pvals_mantel[None, ...], ])
            mantel_pvals_by_reg.append(all_pvals_mantel[None, ...])
            all_scores_mantel = np.concatenate([rnng_scores_mantel[None, ...], lstm_scores_mantel[None, ...], ])
            mantel_scores_by_reg.append(all_scores_mantel[None, ...])
            print(all_pvals_mantel.shape)

            good_scores_rank = all_scores_rank >= 0.15
            print(good_scores_rank.shape)
            bh_threshs = bhy_multiple_comparisons_procedure(all_pvals_mantel)
            print(bh_threshs)
            rnng_good = all_pvals_mantel[0, :] <= bh_threshs[0]
            print(rnng_good.shape)
            lstm_good = all_pvals_mantel[1, :] <= bh_threshs[1]
            good_pvals_mantel = np.concatenate((rnng_good[None, :], lstm_good[None, :]), axis=0)

            win_scores_rank = np.argmax(all_scores_rank, axis=0)
            print(win_scores_rank.shape)
            win_scores_mantel = np.argmin(all_pvals_mantel, axis=0)
            best_scores_mantel = np.max(all_scores_mantel, axis=0)
            rnng_scores_mantel[rnng_scores_mantel != best_scores_mantel] = np.nan
            lstm_scores_mantel[lstm_scores_mantel != best_scores_mantel] = np.nan


            h0 = ax_mantel.plot(time, rnng_scores_mantel)
            h1 = ax_mantel.plot(time, lstm_scores_mantel)
            h0[0].set_label('RNNG')
            h1[0].set_label('LSTM')
            ax_mantel.legend()

            for i_time in range(all_pvals_mantel.shape[-1]):
                if good_pvals_mantel[win_scores_mantel[i_time], i_time]:
                    ax_mantel.scatter(time[i_time], all_scores_mantel[win_scores_mantel[i_time], i_time]+0.05,
                               c=colors[win_scores_mantel[i_time]], linewidths=0.0)
            ax_mantel.set_title(uni_reg[i_reg])
            ax_mantel.set_xlim(args.tmin, args.tmax+0.5)
            ax_mantel.set_xticks(np.arange(args.tmin, args.tmax, 0.5))


            h5 = ax.plot(time, rnng_scores_rank)
            h6 = ax.plot(time, lstm_scores_rank)
            h5[0].set_label('RNNG')
            h6[0].set_label('LSTM')
            ax.legend()

            for i_time in range(all_scores_rank.shape[-1]):
                if good_scores_rank[win_scores_rank[i_time], i_time]:
                    ax.scatter(time[i_time], all_scores_rank[win_scores_rank[i_time], i_time]+0.05,
                               c=colors[win_scores_rank[i_time]], linewidths=0.0)
            ax.set_title(uni_reg[i_reg])
            ax.set_xlim(args.tmin, args.tmax+0.5)
            ax.set_xticks(np.arange(args.tmin, args.tmax, 0.5))
        mantel_scores_by_reg = np.concatenate(mantel_scores_by_reg)
        mantel_scores_by_exp.append(mantel_scores_by_reg[None, ...])
        mantel_pvals_by_reg = np.concatenate(mantel_pvals_by_reg)
        mantel_pvals_by_exp.append(mantel_pvals_by_reg[None, ...])
        kendall_scores_by_reg = np.concatenate(kendall_scores_by_reg)
        kendall_scores_by_exp.append(kendall_scores_by_reg[None, ...])



        max_val = np.max(max_reg)
        min_val = 0.0 #np.min(min_reg)
        for i_reg in range(num_reg):
            axs[i_reg].set_ylim(min_val, max_val+0.1)

        max_val_mantel = np.max(max_reg_mantel)
        min_val_mantel = 0.0 #np.min(min_reg_mantel)
        for i_reg in range(num_reg):
            axs_mantel[i_reg].set_ylim(min_val_mantel, max_val_mantel + 0.1)


        fig.suptitle('{} {} kendall'.format(experiment, args.dist))
        fig.tight_layout()

        fig_mantel.suptitle('{} {} Mantel'.format(experiment, args.dist))
        fig_mantel.tight_layout()

        fig.savefig('RDM_kendall_scores_RvL_{exp}_{metric}_{tmin}_{tmax}_{word}.pdf'.format(exp=experiment, metric=args.dist, tmin=args.tmin, tmax=args.tmax, word=args.word))
        fig_mantel.savefig('RDM_mantel_scores_RvL_{exp}_{metric}_{tmin}_{tmax}_{word}.pdf'.format(exp=experiment, metric=args.dist,
                                                                                    tmin=args.tmin, tmax=args.tmax, word=args.word))
    kendall_scores_by_exp = np.concatenate(kendall_scores_by_exp)
    mantel_scores_by_exp = np.concatenate(mantel_scores_by_exp)
    mantel_pvals_by_exp = np.concatenate(mantel_scores_by_exp)
    print(kendall_scores_by_exp.shape)
    num_reg = kendall_scores_by_exp.shape[1]
    for i_model, model in enumerate(['RNNG', 'LSTM']):
        fig, axs = plt.subplots(num_reg, 1, figsize=(20, 20))
        max_val = np.empty((num_reg,))
        for j_reg in range(num_reg):
            all_reg_scores = np.squeeze(mantel_scores_by_exp[:, j_reg, i_model, :])
            max_val[j_reg] = np.max(all_reg_scores)

            all_reg_pvals = np.squeeze(mantel_pvals_by_exp[:, j_reg, i_model, :])
            win_scores_mantel = np.argmin(all_reg_pvals, axis=0)
            best_scores_mantel = np.max(all_reg_scores, axis=0)
            bh_threshs = bhy_multiple_comparisons_procedure(all_reg_pvals)
            all_reg_good = []
            colors = ['b', 'g', 'r', 'c']
            for k_exp, exp in enumerate(['krns2', 'PassAct2']):
                exp_scores = all_reg_scores[k_exp, :]
                exp_scores[exp_scores != best_scores_mantel] = np.nan
                exp_good_pvals = all_reg_pvals[k_exp, :] <= bh_threshs[k_exp]
                all_reg_good.append(exp_good_pvals)
                h = axs[j_reg].plot(time, exp_scores)
                h[0].set_label(exp)
            axs[j_reg].legend()
            for k_time in range(all_reg_scores.shape[-1]):
                if all_reg_good[win_scores_mantel[k_time]][k_time]:
                    axs[j_reg].scatter(time[k_time], all_reg_scores[win_scores_mantel[k_time], k_time] + 0.05,
                                      c=colors[win_scores_mantel[k_time]], linewidths=0.0)
            axs[j_reg].set_title(uni_reg[j_reg])
            axs[j_reg].set_xlim(args.tmin, args.tmax + 0.5)
            axs[j_reg].set_xticks(np.arange(args.tmin, args.tmax, 0.5))
        max_val = np.max(max_val)
        min_val = 0.0
        for j_reg in range(num_reg):
            axs[j_reg].set_ylim(min_val, max_val + 0.1)
        fig.suptitle('{} {} Mantel'.format(model, args.dist))
        fig.tight_layout()
        fig.savefig('RDM_mantel_scores_KvP_{model}_{metric}_{tmin}_{tmax}_{word}.pdf'.format(model=model, metric=args.dist,
                                                                                   tmin=args.tmin, tmax=args.tmax,
                                                                                   word=args.word))


    plt.show()
