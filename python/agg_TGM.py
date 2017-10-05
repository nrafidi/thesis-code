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
               'alg':['LASSO', 'GNB'],
               'z':['T', 'F'],
               'avg':['T', 'F'],
               'ni':'\d+',
               'nr':'\d+',
               'rsPerm':'\d+',
               'rsCV': '\d+',
               'rsSCV':'\d+'}


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
    if isinstance(PARAM_TYPES[param_name], list):
        for val in PARAM_TYPES[param_name]:
            if param_name + val in fname:
                return val
        raise ValueError('Invalid fname')

    m = re.match('.*_' + param_name + '(\d+)_.*', fname)
    if m:
        return m.group(1)
    else:
        raise ValueError('Invalid fname for param {}'.format(param_name))

def get_param_str(param_name, param_val):
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
    result_dir = run_TGM.SAVE_DIR.format(top_dir=run_TGM.TOP_DIR.format(exp=exp), sub=sub)

    fname = result_dir + 'TGM_' + sub + '_' + sen_type + '_' + word + '_'
    for p in PARAMS_TO_AGG:
        if param_specs is not None:
            if p in param_specs:
                p_val = param_specs[p]
                fname += p + str(p_val) + '_'
            else:
                fname += p + '*_'
        else:
            fname += p + '*_'

    fname += mode + '.npz'
    print(fname)

    result_files = glob.glob(fname)
    i_f = 0
    for f in result_files:
        #print(f)
        print(i_f)
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
        if mode == 'pred':
            tgm = tgm_from_preds(result['preds'], result['l_ints'], result['cv_membership'], accuracy)
            print(tgm.shape)
        else:
            tgm = result['coef']

        sub_results.append(tgm)

        sub_time['time'].append(result['time'])
        sub_time['win_starts'].append(result['win_starts'])

        i_f += 1

    return sub_results, sub_params, sub_time


def get_diag_by_param(result_dict, param_dict, time_dict, param, param_specs, param_limit=None):
    diag_by_sub = []
    param_by_sub = []
    time_by_sub = []
    start_by_sub = []
    for sub in result_dict:
        diag = []
        param_of_interest = [int(p) for p in param_dict[sub][param]]
        tgm_of_interest = result_dict[sub]
        time_of_interest = time_dict[sub]['time']
        
        starts_of_interest = time_dict[sub]['win_starts']
        
        ind_spec = [True] * len(param_of_interest)
        for p in param_specs:
            p_of_interest = np.array(param_dict[sub][p])
            ind_spec = np.logical_and(ind_spec, p_of_interest == str(param_specs[p]))
        tgm_of_interest = list(compress(tgm_of_interest, ind_spec))
        param_of_interest = list(compress(param_of_interest, ind_spec))
        time_of_interest = list(compress(time_of_interest, ind_spec))
        starts_of_interest = list(compress(starts_of_interest, ind_spec))
        sort_inds = np.argsort(np.array(param_of_interest).astype('int'))

        tgm_of_interest = [tgm_of_interest[i] for i in sort_inds]
        param_of_interest = [param_of_interest[i] for i in sort_inds]
        time_of_interest = [np.array(time_of_interest[i]).astype(np.float) for i in sort_inds]
        starts_of_interest = [np.array(starts_of_interest[i]).astype('int') for i in sort_inds]

        if param_limit is not None:
            limit_ind = param_of_interest.index(param_limit) + 1
            tgm_of_interest = tgm_of_interest[:limit_ind]
            param_of_interest = param_of_interest[:limit_ind]
            starts_of_interest = starts_of_interest[:limit_ind]
            time_of_interest = time_of_interest[:limit_ind]

        min_size = 10000
        for tgm in tgm_of_interest:
            if tgm.shape[0] < min_size:
                min_size = tgm.shape[0]
            diag.append(np.diag(tgm))

        diag = [tgm_diag[:min_size] for tgm_diag in diag]
        starts_of_interest = [start_arr[:min_size] for start_arr in starts_of_interest]
        for i, start_arr in enumerate(starts_of_interest):
            print(len(time_of_interest[i]))
            #print(time_of_interest[i])
            time_of_interest[i] = time_of_interest[i][start_arr]
            #print(start_arr)
            #print(time_of_interest[i])

        diag_by_sub.append(np.array(diag))
        param_by_sub.append(np.array(param_of_interest))
        time_by_sub.append(np.array(time_of_interest))
        start_by_sub.append(np.array(starts_of_interest))
    diag = np.array(diag_by_sub)
    param_val = np.array(param_by_sub)
    time = np.array(time_by_sub)
    starts = np.array(start_by_sub)
    print(time.shape)
    return diag, param_val, time, starts


def get_coef_by_param(exp,
                  word,
                  sen_type,
                  accuracy,
                  sub,
                  param_specs):
    overlap = param_specs['o']
    sub_results, sub_params, sub_time = agg_results(exp,
                                                            'coef',
                                                            word,
                                                            sen_type,
                                                            accuracy,
                                                            sub,
                                                            param_specs=param_specs)

    sim_mat = np.empty((len(sub_results), len(sub_results[0]) - 1))
    min_win = 10000
    for i_win, res in enumerate(sub_results):
        win_size = int(sub_params['w'][i_win])
        if len(res) < min_win:
            min_win = len(res)
        for i_coef in range(1, len(res)):
            prev_res = np.zeros(res[i_coef - 1].shape)
            prev_res[res[i_coef - 1] < 0] = -1.0
            prev_res[res[i_coef - 1] > 0] = 1.0

            curr_res = np.zeros(res[i_coef].shape)
            curr_res[res[i_coef] < 0] = -1.0
            curr_res[res[i_coef] > 0] = 1.0

            prev_mat = np.reshape(np.sum(prev_res, axis=0), (win_size, 306))
            curr_mat = np.reshape(np.sum(curr_res, axis=0), (win_size, 306))

            if overlap < prev_mat.shape[0]:
                np.roll(prev_mat, -overlap)
                end_ind = prev_mat.shape[0] - overlap
                prev_mat = prev_mat[:end_ind, :]
                curr_mat = curr_mat[:end_ind, :]

            #         if win_size > overlap:
            #             f, axs = plt.subplots(2, 1)ddd
            #             axs[0].imshow(prev_mat, interpolation='nearest', aspect='auto')
            #             axs[1].imshow(curr_mat, interpolation='nearest', aspect='auto')
            #              plt.show()
            sim_mat[i_win, i_coef - 1] = np.linalg.norm(curr_mat - prev_mat)
    sim_mat = sim_mat[:, 0:(min_win - 1)]
    return sim_mat


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'coef'
    word = 'firstNoun'
    sen_type = 'active'
    accuracy = 'abs'
    sub = 'B'

    param_specs = {'o': 12,
                   'pd': 'F',
                   'pr': 'F',
                   'alg': 'LR',
                   'F': 2,
                   'z': 'F',
                   'avg': 'F',
                   'ni': 2,
                   'nr': 10,
                   'rs': 1}
    sub_results, sub_params, sub_time = agg_results(exp,
                                                    mode,
                                                    word,
                                                    sen_type,
                                                    accuracy,
                                                    sub,
                                                    param_specs=param_specs)
    print(sub_params)
    for i, res in enumerate(sub_results):
        win_size = int(sub_params['w'][i])
        mat = res[1]
        mat = np.sum(mat, axis=0)
        print(mat.shape)
        mat = np.reshape(mat, (win_size, 306))
        f, ax = plt.subplots()
        ax.imshow(mat, interpolation='nearest')
        plt.savefig('mat{}_T_next.png'.format(i), bbox_inches='tight')
        print(mat.shape)
        woof = np.array(np.where(mat > 0))
        print(woof.shape)
        for moo in range(5):
            print('({}, {})'.format(woof[0, moo], woof[1, moo]))
            print(mat[woof[0, moo], woof[1, moo]])
