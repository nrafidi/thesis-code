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
                tgm = sub_results[0]
                diag = np.diag(tgm)
                time = sub_time['time'][0][sub_time['win_starts'][0]]
                print(sub_masks[0].shape)
                    # fig, axs = plt.subplots(1, 2)
                    # h = axs[0].imshow(sub_avg_list[0], interpolation='nearest', vmin=0, vmax=1)
                    # plt.colorbar(h)
                    # axs[1].plot(np.diag(sub_avg_list[0]))
                    # plt.savefig('TGM_overlap{}_GNB-FS_{}_{}_{}.pdf'.format(o, param_specs['w'], word, sen_type))
                    # # plt.show()
