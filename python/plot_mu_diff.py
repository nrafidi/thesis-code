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


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'coef'
    accuracy = 'abs'
    o = 3
    w = -1
    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    for word in ['firstNoun', 'verb', 'secondNoun']:
        for sen_type in ['passive', 'active']:
            for sub in load_data.VALID_SUBS[exp]:
                param_specs = {'o': o,
                               'w': w,
                               'pd': 'F',
                               'pr': 'F',
                               'alg': 'GNB-FS',
                               'F': 2,
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
                mu_diff = sub_results[0][-1][0]
                mu_diff = mu_diff[sorted_inds, :]
                num_time = mu_diff.shape[1]
                fulltime = sub_time['time'][0]
                fulltime[np.abs(fulltime) < 1e-15] = 0
                fulltime = fulltime[:num_time]

                fig, ax = plt.subplots()
                h = ax.imshow(mu_diff, interpolation='nearest', aspect='auto')
                ax.set_yticks(yticks_sens)
                ax.set_yticklabels(uni_reg)
                ax.set_ylabel('Sensors')
                ax.set_xticks(range(0, num_time, 250))
                ax.set_xticklabels(fulltime[::250])
                ax.set_xlabel('Time')
                plt.colorbar(h)
                plt.show()


                # tgm = sub_results[0]
                # print(tgm.shape)
                # diag = np.diag(tgm)
                # print(diag.shape)
                #
                # print(time.shape)
                # num_time = time.shape[0]

                # print(fulltime.shape)
                # num_fulltime = fulltime.shape[0]
                # masks = np.sum(sub_masks[0], axis=0)
                # print(masks.shape)
                # accum_mask = accum_over_time(masks, o)
                # print(accum_mask.shape)
                # accum_mask = accum_mask[:, :num_fulltime]
                # meow = accum_mask
                # accum_mask = accum_mask[sorted_inds, :]
                #

                # ax.set_yticks(range(0, num_time, 25))
                # ax.set_yticklabels(time[::25])
                # ax.set_ylabel('Train Window Start')
                # ax.set_xticks(range(0, num_time, 25))
                # ax.set_xticklabels(time[::25])
                # ax.set_xlabel('Test Window Start')
                # plt.colorbar(h)
                # plt.savefig('TGM_{}_o{}_w{}_{}_{}_GNB-FS.pdf'.format(sub, o, w, word, sen_type))
                #
                # fig, ax = plt.subplots()
                # ax.plot(time, diag)
                # ax.set_ylim([0, 1])
                # ax.set_ylabel('Accuracy')
                # ax.set_xlabel('Train Window Start')
                # plt.savefig('Diag_{}_o{}_w{}_{}_{}_GNB-FS.pdf'.format(sub, o, w, word, sen_type))
                #
                # fig, ax = plt.subplots()
                # h = ax.imshow(accum_mask, interpolation='nearest', aspect='auto', vmin=0, vmax=15)
                # ax.set_yticks(yticks_sens)
                # ax.set_yticklabels(uni_reg)
                # ax.set_ylabel('Sensors')
                # ax.set_xticks(range(0, num_fulltime, 250))
                # ax.set_xticklabels(fulltime[::250])
                # ax.set_xlabel('Time')
                # plt.colorbar(h)
                # plt.savefig('Masks_{}_o{}_w{}_{}_{}_GNB-FS.pdf'.format(sub, o, w, word, sen_type), bbox_inches='tight')
                # # plt.show()
