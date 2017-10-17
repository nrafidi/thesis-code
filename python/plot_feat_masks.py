import matplotlib
matplotlib.use('Agg') # TkAgg - only works when sshing from office machine
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


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'pred'
    accuracy = 'abs'
    o = 12
    w = 100
    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    for word in ['firstNoun', 'verb', 'secondNoun']:
        for sen_type in ['passive', 'active']:
            sub_mask_add = []
            sub_tgm_avg = []
            for sub in load_data.VALID_SUBS[exp]:
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
                print('meow')
                tgm = sub_results[0]
                sub_tgm_avg.append(tgm)
                # print(tgm.shape)
                diag = np.diag(tgm)
                print(np.max(diag))
                time = sub_time['time'][0][sub_time['win_starts'][0]]
                # print(time.shape)
                num_time = time.shape[0]
                fulltime = sub_time['time'][0]
                fulltime[np.abs(fulltime) < 1e-15] = 0
                # print(fulltime.shape)
                num_fulltime = fulltime.shape[0]
                masks = np.sum(sub_masks[0], axis=0)
                print(np.max(masks))
                accum_mask = accum_over_time(masks, o)
                # print(accum_mask.shape)
                accum_mask = accum_mask[:, :num_fulltime]
                print(np.max(accum_mask))
                meow = accum_mask
                accum_mask = accum_mask[sorted_inds, :]
                sub_mask_add.append(accum_mask)

                # fig, ax = plt.subplots()
                # h = ax.imshow(tgm, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
                # ax.set_yticks(range(0, num_time, 25))
                # ax.set_yticklabels(time[::25])
                # ax.set_ylabel('Train Window Start')
                # ax.set_xticks(range(0, num_time, 25))
                # ax.set_xticklabels(time[::25])
                # ax.set_xlabel('Test Window Start')
                # plt.colorbar(h)
                # plt.savefig('TGM_{}_o{}_w{}_{}_{}_{}F_GNB-FS.pdf'.format(sub, o, w, word, sen_type, param_specs['F']))
                #
                # # fig, ax = plt.subplots()
                # # ax.plot(time, diag)
                # # ax.set_ylim([0, 1])
                # # ax.set_ylabel('Accuracy')
                # # ax.set_xlabel('Train Window Start')
                # # plt.savefig('Diag_{}_o{}_w{}_{}_{}_{}F_GNB-FS.pdf'.format(sub, o, w, word, sen_type, param_specs['F']))
                #
                # fig, ax = plt.subplots()
                # h = ax.imshow(accum_mask, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
                # ax.set_yticks(yticks_sens)
                # ax.set_yticklabels(uni_reg)
                # ax.set_ylabel('Sensors')
                # ax.set_xticks(range(0, num_fulltime, 250))
                # ax.set_xticklabels(fulltime[::250])
                # ax.set_xlabel('Time')
                # plt.colorbar(h)
                # plt.savefig('Masks_{}_o{}_w{}_{}_{}_{}F_GNB-FS.pdf'.format(sub, o, w, word, sen_type, param_specs['F']), bbox_inches='tight')
                # # plt.show()
            sub_tgm = np.mean(np.array(sub_tgm_avg), axis=0)
            sub_mask = np.sum(np.array(sub_mask_add), axis=0)

            if w > 0:
                vmax = 100 #(w/o)*len(load_data.VALID_SUBS[exp])*param_specs['F']
            else:
                vmax = len(load_data.VALID_SUBS[exp])*param_specs['F']

            fig, ax = plt.subplots()
            h = ax.imshow(sub_tgm, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            ax.set_yticks(range(0, num_time, 25))
            ax.set_yticklabels(time[::25])
            ax.set_ylabel('Train Window Start')
            ax.set_xticks(range(0, num_time, 25))
            ax.set_xticklabels(time[::25])
            ax.set_xlabel('Test Window Start')
            plt.colorbar(h)
            plt.savefig('TGM_subAvg_o{}_w{}_{}_{}_{}F_GNB-FS.pdf'.format(o, w, word, sen_type, param_specs['F']))

            fig, ax = plt.subplots()
            h = ax.imshow(sub_mask, interpolation='nearest', aspect='auto', vmin=0, vmax=vmax)
            ax.set_yticks(yticks_sens)
            ax.set_yticklabels(uni_reg)
            ax.set_ylabel('Sensors')
            ax.set_xticks(range(0, num_fulltime, 250))
            ax.set_xticklabels(fulltime[::250])
            ax.set_xlabel('Time')
            plt.colorbar(h)
            plt.savefig('Masks_subAvg_o{}_w{}_{}_{}_{}F_GNB-FS.pdf'.format(o, w, word, sen_type, param_specs['F']), bbox_inches='tight')
            # plt.show()
