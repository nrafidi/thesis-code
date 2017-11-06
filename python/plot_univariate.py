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


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


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

    for sen_type in ['active', 'passive']:
        tgm_by_word = []
        for word in ['firstNoun', 'verb', 'secondNoun']:
            tgm_by_sub = []
            for sub in ['B', 'C']: #load_data.VALID_SUBS[exp]:
                param_specs = {'o': o,
                               'w': w,
                               'pd': 'F',
                               'pr': 'F',
                               'alg': 'GNB-FS',
                               'F': 32,
                               'z': 'F',
                               'avg': 'F',
                               'ni': 2,
                               'nr': 10,
                               'rsPerm': 1,
                               'rsCV': run_TGM.CV_RAND_STATE,
                               'rsSCV': run_TGM.SUB_CV_RAND_STATE}
                sub_results, _, sub_time, sub_masks = agg_TGM.agg_results(exp,
                                                                          mode,
                                                                          word,
                                                                          sen_type,
                                                                          accuracy,
                                                                          sub,
                                                                          param_specs=param_specs)
                tgm = sub_results[0]
                if sens != 'comb' and sens != 'reg' and sens != 'wb':
                    tgm = tgm[:, :, sorted_inds, :]
                if sens == 'avg' or sens == 'max':
                    tgm = comb_by_loc(tgm, sens)
                tgm_by_sub.append(tgm)

            concat_tgm = np.squeeze(np.concatenate(tgm_by_sub))
            print(concat_tgm.shape)
            if sens == 'wb':
                concat_tgm = np.reshape(concat_tgm, (concat_tgm.shape[0], 1, concat_tgm.shape[1]))
            (num_sub, num_sens, num_time) = concat_tgm.shape
            print(num_sub)
            print(num_sens)

            if word == 'firstNoun' and sen_type == 'active':
                word_tgm = concat_tgm[:, :, :(num_time-250)]
            elif word == 'firstNoun' and sen_type == 'passive':
                word_tgm = concat_tgm[:, :, :(num_time-750)]
            elif word == 'verb' and sen_type == 'active':
                word_tgm = np.concatenate((np.zeros((num_sub, num_sens, 250)), concat_tgm[:, :, :(num_time-250)]), axis=2)
            elif word == 'verb' and sen_type == 'passive':
                word_tgm = np.concatenate((np.zeros((num_sub, num_sens, 500)), concat_tgm[:, :, :(num_time-750)]), axis=2)
            elif word == 'secondNoun' and sen_type == 'active':
                word_tgm = np.concatenate((np.zeros((num_sub, num_sens, 750)), concat_tgm[:, :, :(num_time-250)]), axis=2)
            else:
                word_tgm = np.concatenate((np.zeros((num_sub, num_sens, 1250)), concat_tgm[:, :, :(num_time-1250)]), axis=2)
            print(word)
            print(word_tgm.shape)
            tgm_by_word.append(word_tgm[None, ...])


        word_tgm = np.concatenate(tgm_by_word)
        print(word_tgm.shape)
        avg_tgm = np.mean(word_tgm, axis=1)
        print(avg_tgm.shape)
        total_best = np.zeros((word_tgm.shape[2], word_tgm.shape[3], 3))

        for i in range(word_tgm.shape[2]):
            for j in range(word_tgm.shape[3]):
                sub_perf = np.sum(word_tgm[:, :, i, j] >= 0.25, axis=1)
                if np.any(sub_perf > 1):
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
                h0 = ax.plot(avg_tgm[0, i, :], c=colors[0])
                h1 = ax.plot(avg_tgm[1, i, :], c=colors[1])
                h2 = ax.plot(avg_tgm[2, i, :], c=colors[2])
                h0[0].set_label('firstNoun')
                h1[0].set_label('verb')
                h2[0].set_label('secondNoun')
                num_time = total_best.shape[1]

                for j in range(num_time):
                    if total_best[i, j, 0] != 0.8:
                        best_word = np.where(np.squeeze(total_best[i, j, :]))
                        # print(best_word[0][0])
                        ax.scatter(j, avg_tgm[best_word[0][0], i, j] + 0.1, c=colors[best_word[0][0]], linewidths=0.0)

                time = np.arange(0.0, 4.5, 0.002)
                ax.set_xlim(0, num_time + 500)
                ax.set_ylim(0, 0.8)
                ax.set_xticks(range(0, num_time, 250))
                ax.set_xticklabels(time[::250])
                ax.legend()
                if sens == 'reg':
                    ax.set_title(uni_reg[i])
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
    plt.show()


