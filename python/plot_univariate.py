import argparse
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import load_data
from scipy.stats.mstats import zscore
import scipy.io as sio
import agg_TGM
import run_TGM
import coef_sim
from scipy.stats import norm
from scipy import stats
import warnings
import os
from scipy.stats import kendalltau


SENSOR_MAP = '/home/nrafidi/sensormap.mat'
PERM_FILE = '/share/volume0/nrafidi/{exp}_TGM/{sub}/TGM_{sub}_{sen_type}_{word}_w{win_len}_o{overlap}_pd{pdtw}_pr{perm}_F{num_folds}_alg{alg}_' \
            'z{zscore}_avg{doAvg}_ni{inst}_nr{rep}_rsPerm{rsPmin}-{rsPmax}_agg{accuracy}_rsCV{rsC}_rsSCV{rsS}_{mode}_lp'


def accum_over_sub(sub_results):
    subjects = [sub_key for sub_key in sub_results.keys() if sub_results[sub_key]]
    num_param = len(sub_results[subjects[0]])
    result_by_param = []
    for i_param in range(num_param):
        result_by_sub = []
        for sub in subjects:result_by_sub.append(sub_results[sub][i_param])
        result_by_param.append(np.sum(np.array(result_by_sub), axis=0))
    return result_by_param


def accum_over_time(masks, overlap):
    num_time = masks.shape[0]*overlap
    (num_sensors, win_len) = masks[0].shape
    accum_mask = np.zeros((num_sensors, num_time + win_len))
    for t in range(masks.shape[0]):
        start_ind = t*overlap
        end_ind = start_ind + win_len
        accum_mask[:, start_ind:end_ind] += masks[t]
    return accum_mask


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg

def comb_by_loc(tgm, sens):
    (s0, s1, s2, s3) = tgm.shape
    new_tgm = np.empty((s0, s1, s2/3, s3))
    i_new = 0
    for i in range(0, s2, 3):
        moo = tgm[:, :, i:(i+3), :]
        assert moo.shape[2] == 3
        if sens == 'avg':
            new_tgm[:, :, i_new, :] = np.mean(tgm[:, :, i:(i+3), :], axis=2)
        else:
            new_tgm[:, :, i_new, :] = np.max(tgm[:, :, i:(i + 3), :], axis=2)
        i_new += 1
    return new_tgm


def correct_pvals(uncorrected_pvals):
    print('moo')
    up_shape = uncorrected_pvals.shape
    print(up_shape)
    new_pvals = np.empty((uncorrected_pvals.shape[1], uncorrected_pvals.shape[2]))
    print(new_pvals.shape)
    for i in range(uncorrected_pvals.shape[1]):
        for j in range(uncorrected_pvals.shape[2]):
            dist_over_sub = uncorrected_pvals[:, i, j]
            dist_over_sub[dist_over_sub >= 1.0] = 1.0 - 1e-15
            meow = norm.ppf(dist_over_sub)
            if np.any(np.isinf(meow)):
                print('Inf')
                print(dist_over_sub)
            if np.any(np.isnan(meow)):
                print('NaN')
                print(dist_over_sub)

            # if np.std(meow) == 0.0:
            #     meow[0] += 1e-15

            t_stat, new_pvals[i, j] = stats.ttest_1samp(meow, 0.0)
            if t_stat < 0.0:
                new_pvals[i, j] /= 2.0
            else:
                new_pvals[i, j] = 1.0 - new_pvals[i, j]/2.0
            assert not np.isnan(new_pvals[i, j])
            assert not np.isinf(new_pvals[i, j])
    bh_thresh = bhy_multiple_comparisons_procedure(new_pvals)

    corr_pvals = new_pvals <= bh_thresh[:, None]
    return corr_pvals, new_pvals


def bhy_multiple_comparisons_procedure(uncorrected_pvalues, alpha=0.01):
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
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--sensors', default='all')
    parser.add_argument('--sen_type', default='active')
    parser.add_argument('--word', default='firstNoun')
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    args = parser.parse_args()
    # warnings.filterwarnings('error')
    exp = args.experiment
    mode = 'uni'
    sens = args.sensors


    o = 3
    w = -1
    sorted_inds, sorted_reg = sort_sensors()

    # mags
    if sens == 'mags':
        sorted_inds = sorted_inds[2::3]
        sorted_reg = sorted_reg[2::3]
        accuracy = 'abs'
    elif sens == 'grad1':
        sorted_inds = sorted_inds[0::3]
        sorted_reg = sorted_reg[0::3]
        accuracy = 'abs'
    elif sens == 'grad2':
        sorted_inds = sorted_inds[1::3]
        sorted_reg = sorted_reg[1::3]
        accuracy = 'abs'
    elif sens == 'all':
        accuracy = 'abs'
    elif sens == 'avg' or sens == 'max':
        sorted_reg = sorted_reg[0::3]
        accuracy = 'abs'
    elif sens == 'comb':
        sorted_reg = sorted_reg[0::3]
        accuracy = 'abs-sens'

    uni_reg = np.unique(sorted_reg)

    if sens == 'reg':
        yticks_sens = range(uni_reg.size)
        accuracy = 'abs-reg'
    elif sens == 'wb':
        accuracy = 'abs-wb'
    else:
        yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    avg_by_sen_type = []
    masked_avg_by_sen_type = []
    for sen_type in ['active', 'passive']:
        tgm_by_word = []
        corr_p_by_word = []
        scatter_tgm_by_word = []
        for word in ['firstNoun', 'verb', 'secondNoun']:
            tgm_by_sub = []
            pval_by_sub = []
            for sub in ['I', 'D', 'A']: #load_data.VALID_SUBS[exp]:
                param_specs = {'o': o,
                               'w': w,
                               'pd': 'F',
                               'pr': 'T',
                               'alg': 'GNB-FS',
                               'F': 32,
                               'z': 'F',
                               'avg': 'F',
                               'ni': 2,
                               'nr': 10,
                               'rsCV': run_TGM.CV_RAND_STATE,
                               'rsSCV': run_TGM.SUB_CV_RAND_STATE}

                perm_agg_file = PERM_FILE.format(exp=exp, sub=sub, sen_type=sen_type, word=word, win_len=w, overlap=o,
                                                 pdtw=param_specs['pd'], perm='T', num_folds=param_specs['F'],
                                                 alg=param_specs['alg'], zscore=param_specs['z'], doAvg=param_specs['avg'],
                                                 inst=param_specs['ni'], rep=param_specs['nr'], rsPmin=1, rsPmax=99,
                                                 accuracy=accuracy,rsC=param_specs['rsCV'], rsS=param_specs['rsSCV'],mode=mode)

                if os.path.isfile(perm_agg_file + '.npz'):
                    result = np.load(perm_agg_file + '.npz')
                    perm_tgm = result['perm_tgm']
                else:
                    print(perm_agg_file)
                    sub_perm_results, _, _, _ = agg_TGM.agg_results(exp,
                                                                    mode,
                                                                    word,
                                                                    sen_type,
                                                                    accuracy,
                                                                    sub,
                                                                    param_specs=param_specs)
                    perm_tgm = np.stack(sub_perm_results)
                    np.savez_compressed(perm_agg_file,
                                        perm_tgm=perm_tgm)

                param_specs['rsPerm'] = 1
                param_specs['pr'] = 'F'
                sub_results, _, sub_time, sub_masks = agg_TGM.agg_results(exp,
                                                                          mode,
                                                                          word,
                                                                          sen_type,
                                                                          accuracy,
                                                                          sub,
                                                                          param_specs=param_specs)
                tgm = sub_results[0]

                print('meow')
                print(perm_tgm.shape)
                print(tgm.shape)
                if sens != 'comb' and sens != 'reg' and sens != 'wb':
                    tgm = tgm[:, :, sorted_inds, :]
                    perm_tgm = perm_tgm[:, :, :, sorted_inds, :]
                if sens == 'avg' or sens == 'max':
                    tgm = comb_by_loc(tgm, sens)
                tgm_by_sub.append(tgm)
                print('perm code')
                perms_greater = perm_tgm >= tgm[None, ...]
                print(perms_greater.shape)
                sum_perms_greater = np.sum(perms_greater, axis=0)
                print(sum_perms_greater.shape)
                print(np.min(sum_perms_greater))
                pvals = (sum_perms_greater + 1.0)/float(perms_greater.shape[0])
                print(np.min(pvals))
                print('woof')
                print(pvals.shape)
                pval_by_sub.append(pvals[None, :])

            concat_tgm = np.squeeze(np.concatenate(tgm_by_sub))
            print('oink')
            print(concat_tgm.shape)
            total_pvals = np.squeeze(np.concatenate(pval_by_sub))
            print(total_pvals.shape)

            if sens == 'wb':
                concat_tgm = np.reshape(concat_tgm, (concat_tgm.shape[0], 1, concat_tgm.shape[1]))
                total_pvals = np.reshape(total_pvals, (total_pvals.shape[0], 1, total_pvals.shape[1]))
            corr_pvals, new_pvals = correct_pvals(total_pvals)

            fig, ax = plt.subplots()
            h = ax.imshow(total_pvals[2, ...], interpolation='nearest', aspect='auto', vmin=0.0, vmax=0.05)
            plt.colorbar(h)
            fig.suptitle('Single Subject p values {} {}'.format(sen_type, word))

            for i in range(3):
                ktau, _ = kendalltau(total_pvals[i, ...], new_pvals)
                print(ktau)

            fig, ax = plt.subplots()
            h = ax.imshow(new_pvals, interpolation='nearest', aspect='auto', vmin=0.0, vmax=0.05)
            plt.colorbar(h)
            fig.suptitle('Combined values {} {}'.format(sen_type, word))

            fig, ax = plt.subplots()
            h = ax.imshow(corr_pvals, interpolation='nearest', aspect='auto', vmin=0.0, vmax=0.05)
            plt.colorbar(h)
            fig.suptitle('Surviving p values {} {}'.format(sen_type, word))

            # plt.show()

            print(np.sum(corr_pvals))



            (num_sub, num_sens, num_time) = concat_tgm.shape
            print(num_sub)
            print(num_sens)



            if word == 'secondNoun' and sen_type == 'passive':
                word_tgm = np.concatenate((concat_tgm,
                                           np.zeros((num_sub, num_sens, 1))),
                                          axis=2)
                corr_pvals = np.concatenate((corr_pvals,
                                           np.ones((num_sens, 1))),
                                          axis=1)
            else:
                word_tgm = concat_tgm
            tgm_for_scatter = np.copy(word_tgm)
            for i_sub in range(num_sub):
                meow = tgm_for_scatter[i_sub, :, :]
                meow[np.logical_not(corr_pvals)] = np.nan
                tgm_for_scatter[i_sub, :, :] = meow
            # print(tgm_for_scatter)
            print(word)
            print(word_tgm.shape)
            tgm_by_word.append(word_tgm[None, ...])

            print(tgm_for_scatter.shape)
            scatter_tgm_by_word.append(tgm_for_scatter[None, ...])
            corr_p_by_word.append(corr_pvals)


        word_tgm = np.concatenate(tgm_by_word)
        woof = np.concatenate(scatter_tgm_by_word)
        print('ahoy')
        word_scatter = np.mean(woof, axis=1)
        print(word_scatter)
        print(np.any(np.isnan(word_scatter)))



        print(word_tgm.shape)
        avg_tgm = np.mean(word_tgm, axis=1)
        avg_by_sen_type.append(avg_tgm[None, ...])
        best_avg = np.max(avg_tgm, axis=0)
        masked_avg_tgm = np.copy(avg_tgm)
        masked_avg_tgm[masked_avg_tgm != best_avg] = np.nan
        masked_avg_by_sen_type.append(masked_avg_tgm[None, ...])

        print(avg_tgm.shape)
        total_best = np.zeros((word_tgm.shape[2], word_tgm.shape[3], 3))

        for i in range(word_tgm.shape[2]):
            for j in range(word_tgm.shape[3]):
                sub_perf = np.sum(word_tgm[:, :, i, j] >= 0.3, axis=1)
                if np.any(sub_perf > 4):
                    total_best[i, j, np.argmax(avg_tgm[:, i, j])] = 1
                else:
                    total_best[i, j, :] = [0.8, 0.8, 0.8]


        if sens == 'reg' or sens == 'wb':
            fig, axs = plt.subplots(avg_tgm.shape[1], 1, figsize=(20, 20))
            colors = ['r', 'g', 'b']
            for i in range(avg_tgm.shape[1]):
                if sens == 'wb':
                    ax = axs
                else:
                    ax = axs[i]
                h0 = ax.plot(masked_avg_tgm[0, i, :], c=colors[0])
                h1 = ax.plot(masked_avg_tgm[1, i, :], c=colors[1])
                h2 = ax.plot(masked_avg_tgm[2, i, :], c=colors[2])
                h0[0].set_label('firstNoun')
                h1[0].set_label('verb')
                h2[0].set_label('secondNoun')
                num_time = total_best.shape[1]

                # print(word_scatter.shape)
                for k in range(2, -1, -1):
                    for j in range(num_time):
                        if not np.isnan(word_scatter[k, i, j]):
                            ax.scatter(j, 0.7, c=colors[k], linewidths=0.0)

                time = np.arange(0.0, 4.5, 0.002)
                ax.set_xlim(0, num_time + 500)
                ax.set_ylim(0.2, 0.8)
                ax.set_xticks(range(0, num_time, 250))
                ax.set_xticklabels(time[::250])
                ax.legend()
                if sens == 'reg':
                    ax.set_title(uni_reg[i])
            fig.suptitle(sen_type)
            fig.tight_layout()
        else:
            fig, ax = plt.subplots()
            ax.imshow(total_best, interpolation='nearest', aspect='auto')
            num_time = total_best.shape[1]
            time = np.arange(0.0, 4.5, 0.002)
            ax.set_xticks(range(0, num_time, 250))
            ax.set_xticklabels(time[::250])
            ax.set_yticks(yticks_sens)
            ax.set_yticklabels(uni_reg)
            ax.set_ylabel('Sensors')
            ax.set_title(sen_type)
        fig.savefig('Univariate_NVN_{}_{}_{}_fullperm.pdf'.format(sens, sen_type, args.experiment))

    avg = np.concatenate(avg_by_sen_type, axis=0)
    masked_avg = np.concatenate(masked_avg_by_sen_type, axis=0)
    num_time = masked_avg.shape[-1]
    if sens == 'reg' or sens == 'wb':
        for alignment in ['firstNoun', 'verb', 'secondNoun']:
            if alignment == 'firstNoun':
                active_inds = range(num_time)
                passive_inds = range(num_time)
            elif alignment == 'verb':
                active_inds = range(0, num_time-250)
                passive_inds = range(251, num_time)
            else:
                active_inds = range(0, num_time - 500)
                passive_inds = range(501, num_time)

            fig, axs = plt.subplots(avg.shape[2], 1, figsize=(20, 20))
            colors = ['r', 'g', 'b', 'm', 'y', 'c']
            for i in range(avg.shape[2]):
                if sens == 'wb':
                    ax = axs
                else:
                    ax = axs[i]
                h00 = ax.plot(masked_avg[0, 0, i, active_inds], c=colors[0])
                h01 = ax.plot(masked_avg[0, 1, i, active_inds], c=colors[1])
                h02 = ax.plot(masked_avg[0, 2, i, active_inds], c=colors[2])
                h10 = ax.plot(masked_avg[1, 0, i, passive_inds]+ 0.25, c=colors[3])
                h11 = ax.plot(masked_avg[1, 1, i, passive_inds]+ 0.25, c=colors[4])
                h12 = ax.plot(masked_avg[1, 2, i, passive_inds]+ 0.25, c=colors[5])

                h00[0].set_label('firstNoun active')
                h01[0].set_label('verb active')
                h02[0].set_label('secondNoun active')
                h10[0].set_label('firstNoun passive')
                h11[0].set_label('verb passive')
                h12[0].set_label('secondNoun passive')

                #
                # for j in range(num_time):
                #     if total_best[i, j, 0] != 0.8:
                #         best_word = np.where(np.squeeze(total_best[i, j, :]))
                #         ax.scatter(j, 0.5, c=colors[best_word[0][0]], linewidths=0.0)

                time = np.arange(0.0, 4.5, 0.002)
                ax.set_xlim(0, num_time + 500)
                ax.set_ylim(0.2, 0.8)
                ax.set_xticks(range(0, num_time, 250))
                ax.set_xticklabels(time[::250])
                ax.legend()
                if sens == 'reg':
                    ax.set_title(uni_reg[i])
            fig.suptitle(alignment)
            fig.tight_layout()
            fig.savefig('Univariate_NVN_{}_AvP_{}_{}_perm.pdf'.format(sens, alignment, args.experiment))
    plt.show()


