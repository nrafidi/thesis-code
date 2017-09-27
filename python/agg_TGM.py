import glob
from itertools import compress
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import run_TGM


PARAMS_TO_AGG = ['w', 'o', 'pd', 'pr', 'F', 'alg', 'z', 'avg', 'ni', 'nr', 'rs']
PARAM_TYPES = {'w':'\d+',
               'o':'\d+',
               'pd':['T', 'F'],
               'pr':['T', 'F'],
               'F':'\d+',
               'alg':['LR', 'GNB'],
               'z':['T', 'F'],
               'avg':['T', 'F'],
               'ni':'\d+',
               'nr':'\d+',
               'rs':'\d+'}


def which_word(fname):
    if 'firstNoun' in fname:
        return 'firstNoun'
    elif 'verb' in fname:
        return 'verb'
    elif 'secondNoun' in fname:
        return 'secondNoun'
    else:
        raise ValueError('Invalid fname')


def which_sen_type(fname):
    if 'active' in fname:
        return 'active'
    elif 'passive' in fname:
        return 'passive'
    else:
        raise ValueError('Invalid fname')


def extract_param(param_name, fname):
    # print(param_name)
    if param_name == 'alg':
        for alg in PARAM_TYPES[param_name]:
            if alg in fname:
                return alg
        raise ValueError('Invalid fname')
    if param_name == 'F':
        m = re.match('.*_(\d+)F_.*', fname)
        if m:
            return m.group(1)
        else:
            raise ValueError('Invalid fname')
    if isinstance(PARAM_TYPES[param_name], list):
        for val in PARAM_TYPES[param_name]:
            if param_name + val in fname:
                return val
        raise ValueError('Invalid fname')

    match_str = '_{}(\d+)_'.format(param_name)
    m = re.match('.*_' + param_name + '(\d+)_.*', fname)
    if m:
        return m.group(1)
    else:
        raise ValueError('Invalid fname')

def get_param_str(param_name, param_val):
    if param_name == 'alg':
        return param_val
    if param_name == 'F':
        return '_{}F_'.format(param_val)
    if isinstance(PARAM_TYPES[param_name], list):
        return param_name + param_val

    return '_{}{}_'.format(param_name, param_val)


def tgm_from_preds(preds, l_ints, cv_membership, accuracy='abs'):
    num_folds = preds.shape[0]
    num_win = preds.shape[1]

    if accuracy == 'abs':
        tgm_corr = np.zeros((num_win, num_win))
        tgm_total = np.zeros((num_win, num_win))
        for fold in range(num_folds):
            labels = l_ints[cv_membership[fold, :]]
            # print(labels)
            for i_win in range(num_win):
                for j_win in range(num_win):
                    yhat = np.argmax(preds[fold, i_win, j_win], axis=1)
                    tgm_corr[i_win, j_win] += np.sum(yhat == labels)
                    tgm_total[i_win, j_win] += cv_membership.shape[1]
        tgm = np.divide(tgm_corr, tgm_total)
        return tgm
    else:
        raise ValueError('Not implemented yet')


def agg_results(exp, mode, word, sen_type, accuracy, sub, param_specs=None):
    sub_results = []
    sub_params = {}
    sub_time = {}
    result_dir = run_TGM.SAVE_DIR.format(exp=exp, sub=sub)

    fname = result_dir + 'TGM_' + sub + '_' + sen_type + '_' + word + '_'
    for p in PARAMS_TO_AGG:
        if param_specs is not None:
            if p in param_specs:
                p_val = param_specs[p]
                if p == 'alg':
                    fname += p_val + '_'
                elif p == 'F':
                    fname += str(p_val) + p + '_'
                else:
                    fname += p + str(p_val) + '_'
            else:
                if p == 'alg':
                    fname += '*_'
                elif p == 'F':
                    fname += '*F_'
                else:
                    fname += p + '*_'
        else:
            if p == 'alg':
                fname += '*_'
            elif p == 'F':
                fname += '*F_'
            else:
                fname += p + '*_'

    fname += mode + '.npz'
    print(fname)

    result_files = glob.glob(fname)
    i_f = 0
    for f in result_files:
        if i_f > 5:
            break
        elif i_f > 2:
            print(f)
            for param in PARAMS_TO_AGG:
                if param not in sub_params:
                    sub_params[param] = []
            if 'time' not in sub_time:
                sub_time['time'] = []
                sub_time['win_starts'] = []

            for param in PARAMS_TO_AGG:
                if param not in param_specs:
                    param_val = extract_param(param, f)
                    sub_params[param].append(param_val)

            result = np.load(f)
            tgm = tgm_from_preds(result['preds'], result['l_ints'], result['cv_membership'], accuracy)

            sub_results.append(tgm)

            sub_time['time'].append(result['time'])
            sub_time['win_starts'].append(result['win_starts'])

        i_f += 1

    return sub_results, sub_params, sub_time


def get_diag_by_param(result_dict, param_dict, time_dict, param, param_specs):
    diag_by_sub = []
    param_by_sub = []
    time_by_sub = []
    for sub in result_dict:
        diag = []
        param_of_interest = param_dict[sub][param]
        tgm_of_interest = result_dict[sub]
        time_of_interest = time_dict[sub]['win_starts']
        ind_spec = [True] * len(param_of_interest)
        for p in param_specs:
            p_of_interest = np.array(param_dict[sub][p])
            ind_spec = np.logical_and(ind_spec, p_of_interest == str(param_specs[p]))
        tgm_of_interest = list(compress(tgm_of_interest, ind_spec))
        param_of_interest = list(compress(param_of_interest, ind_spec))
        time_of_interest = list(compress(time_of_interest, ind_spec))
        sort_inds = np.argsort(param_of_interest)

        tgm_of_interest = [tgm_of_interest[i] for i in sort_inds]
        param_of_interest = [param_of_interest[i] for i in sort_inds]
        time_of_interest = [np.array(time_of_interest[i]) for i in sort_inds]

        min_size = 1000
        for tgm in tgm_of_interest:
            if tgm.shape[0] < min_size:
                min_size = tgm.shape[0]
            diag.append(np.diag(tgm))

        diag = [tgm_diag[:min_size] for tgm_diag in diag]
        time_of_interest = [time_arr[:min_size] for time_arr in time_of_interest]

        diag_by_sub.append(np.array(diag))
        param_by_sub.append(np.array(param_of_interest))
        time_by_sub.append(np.array(time_of_interest))
    diag = np.array(diag_by_sub)
    param_val = np.array(param_by_sub)
    time = np.array(time_by_sub)
    return diag, param_val, time


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'pred'
    accuracy = 'abs'
    sub = 'B'
    param_specs = {'w': 12,
                   'o': 12,
                   'pd': 'F',
                   'pr': 'F',
                   'F': 2,
                   'alg': 'LR',
                   'z': 'F',
                   'avg': 'F',
                   'ni': 2,
                   'nr': 10,
                   'rs': 1}
    agg_results(exp, mode, accuracy, sub, param_specs=param_specs)

