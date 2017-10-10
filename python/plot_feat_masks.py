import matplotlib
matplotlib.use('Agg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import load_data
from scipy.stats.mstats import zscore
import agg_TGM
import run_TGM
import coef_sim


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
    for t in range(num_time):
        start_ind = t*overlap
        end_ind = start_ind + win_len
        accum_mask[:, start_ind:end_ind] += masks[t]
    return accum_mask



if __name__ == '__main__':
    exp = 'krns2'
    mode = 'pred'
    accuracy = 'abs'
    o = 12
    w = 100


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
                print('meow')
                tgm = sub_results[0]
                print(tgm.shape)
                diag = np.diag(tgm)
                print(diag.shape)
                time = sub_time['time'][0][sub_time['win_starts'][0]]
                print(time.shape)
                fulltime = sub_time['time'][0]
                print(fulltime.shape)
                masks = np.sum(sub_masks[0], axis=0)
                print(masks.shape)
                accum_mask = accum_over_time(masks, o)
                print(accum_mask.shape)
                accum_mask = accum_mask[:, :fulltime.shape[0]]

                fig, axs = plt.subplots(3, 1)
                h = axs[0].imshow(tgm, interpolation='nearest', vmin=0, vmax=1)
                axs[0].set_yticks(time)
                axs[0].set_yticklabels(time[::25])
                axs[0].set_ylabel('Train Window Start')
                axs[0].set_xticks(time)
                axs[0].set_xticklabels(time[::25])
                axs[0].set_xlabel('Test Window Start')
                plt.colorbar(h)
                axs[1].plot(time, diag)
                axs[1].set_ylim([0, 1])
                axs[1].set_ylabel('Accuracy')
                axs[1].set_xlabel('Train Window Start')
                h = axs[2].imshow(accum_mask, interpolation='nearest', vmin=0, vmax=20)
                axs[2].set_yticks(range(306))
                axs[2].set_yticklabels(range(0, 306, 25))
                axs[2].set_ylabel('Sensors')
                axs[2].set_xticks(fulltime)
                axs[2].set_xticklabels(fulltime[::25])
                axs[2].set_xlabel('Time')
                plt.colorbar(h)
                plt.savefig('TGM_coef_{}_o{}_w{}_{}_{}_GNB-FS.pdf'.format(sub, o, w, word, sen_type))
                # plt.show()
