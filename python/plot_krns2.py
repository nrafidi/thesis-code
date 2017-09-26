import matplotlib.pyplot as plt
import numpy as np
import load_data
import agg_TGM

def get_diag_by_param(result_dict, param_dict, time_dict, sen_type, word, param, param_specs):
    diag_by_sub = []
    param_by_sub = []
    time_by_sub = []
    for sub in result_dict:
        diag = []
        param_of_interest = param_dict[sub][sen_type][word][param]
        tgm_of_interest = result_dict[sub][sen_type][word]
        time_of_interest = time_dict[sub][sen_type][word]['win_starts']

        ind_spec = [True] * len(param_of_interest)
        for p in param_specs:
            p_of_interest = param_dict[sub][sen_type][word][p]
            ind_spec = ind_spec and p_of_interest == param_specs[p]
        tgm_of_interest = tgm_of_interest[ind_spec]
        param_of_interest = param_of_interest[ind_spec]
        time_of_interest = time_of_interest[ind_spec]

        sort_inds = np.argsort(param_of_interest)
        tgm_of_interest = tgm_of_interest[sort_inds]
        param_of_interest = param_of_interest[sort_inds]
        time_of_interest = time_of_interest[sort_inds]

        for tgm in tgm_of_interest:
            diag.append(np.diag(tgm))

        meow = np.array(diag)
        print(meow.shape)

        diag_by_sub.append(diag)
        param_by_sub.append(param_of_interest)
        time_by_sub.append(time_of_interest)


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'pred'
    accuracy = 'abs'
    sub_results = {}
    sub_params = {}
    sub_time = {}
    for sub in ['B']:  # load_data.VALID_SUBS[exp]:
        param_specs = {'o': 12,
                       'pd': 'F',
                       'pr': 'F',
                       'alg': 'LR',
                       'z': 'F',
                       'avg': 'F',
                       'ni': 2,
                       'nr': 10,
                       'rs': 1}
        sub_results[sub], sub_params[sub], sub_time[sub] = agg_TGM.agg_results(exp,
                                                                               mode,
                                                                               accuracy,
                                                                               sub,
                                                                               param_specs=param_specs)
    get_diag_by_param(sub_results, sub_params, sub_time, 'active', 'firstNoun', 'w', {'F': 2})
