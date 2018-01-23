import argparse
import itertools
import matplotlib
matplotlib.use('TkAgg')
import load_data_ordered as load_data
import matplotlib.pyplot as plt
import numpy as np
import run_OH_Reg_LOSO
import scipy.io as sio
from sklearn.metrics import explained_variance_score
import os.path


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

NUM_FEATS = {'krns2': 16,
             'PassAct2': 12,
             'PassAct3': 12}


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['krns2', 'PassAct2', 'PassAct3'], default='krns2')
    parser.add_argument('--sen_type', choices=['active', 'passive'], default='active')
    parser.add_argument('--word', choices=['noun1', 'verb', 'noun2'], default='noun1')
    parser.add_argument('--adjX', choices=[None, 'mean_center', 'zscore'], default=None)
    parser.add_argument('--adjY', choices=[None, 'mean_center', 'zscore'], default=None)
    parser.add_argument('--avgT', action='store_true')
    parser.add_argument('--alg', choices=['ols', 'ridge'], default='ols')
    parser.add_argument('--num_instances', type=int, default=1)
    args = parser.parse_args()

    exp = args.experiment
    subjects = load_data.VALID_SUBS[exp]
    num_feats_all = NUM_FEATS[exp]
    sen_type = args.sen_type
    word = args.word
    adjX = args.adjX
    adjY = args.adjY
    avgT = run_OH_Reg_LOSO.bool_to_str(args.avgT)
    alg = args.alg
    num_instances = args.num_instances

    top_dir = run_OH_Reg_LOSO.TOP_DIR.format(exp=exp)

    total_est_all = []
    total_true_all = []
    total_est_word = []
    total_true_word = []
    loaded_subs = 0.0
    for subject in subjects:
        save_dir = run_OH_Reg_LOSO.SAVE_DIR.format(top_dir=top_dir, sub=subject)
        fname = run_OH_Reg_LOSO.SAVE_FILE.format(dir=save_dir,
                                                 sub=subject,
                                                 sen_type=sen_type,
                                                 word='all',
                                                 perm='F',
                                                 alg=alg,
                                                 adjX=adjX,
                                                 adjY=adjY,
                                                 avgT=avgT,
                                                 inst=num_instances,
                                                 rep=10,
                                                 rsP=1)
        if os.path.isfile(fname + '.npz'):
            result = np.load(fname + '.npz')
            total_est_all.append(result['preds'])
            total_true_all.append(result['test_data_all'])
            loaded_subs += 0.5

        fname = run_OH_Reg_LOSO.SAVE_FILE.format(dir=save_dir,
                                                 sub=subject,
                                                 sen_type=sen_type,
                                                 word=word,
                                                 perm='F',
                                                 alg=alg,
                                                 adjX=adjX,
                                                 adjY=adjY,
                                                 avgT=avgT,
                                                 inst=num_instances,
                                                 rep=10,
                                                 rsP=1)
        if os.path.isfile(fname + '.npz'):
            result = np.load(fname + '.npz')
            total_est_word.append(result['preds'])
            total_true_word.append(result['test_data_all'])
            loaded_subs += 0.5

    print('{} out of {} subjects loaded'.format(loaded_subs, len(subjects)))
    total_est_all = np.concatenate(total_est_all, axis=0)
    total_true_all = np.concatenate(total_true_all, axis=0)
    total_est_word = np.concatenate(total_est_word, axis=0)
    total_true_word = np.concatenate(total_true_word, axis=0)

    num_samples = total_est_all.shape[0]
    num_outputs = total_est_all.shape[-1]

    r2_all = explained_variance_score(total_true_all, total_est_all, multioutput='raw_values')
    r2_all_adj = np.ones((num_outputs,)) - (np.ones((num_outputs,)) - r2_all)*np.divide(float(num_samples - 1),
                                                                                        float(num_samples-num_feats_all-1))

    print('Pre adj r2all = {}'.format(r2_all[0]))
    print('Divisor = {}'.format(np.divide(float(num_samples - 1), float(num_samples - num_feats_all - 1))))
    print('Post adj r2all = {}'.format(r2_all_adj[0]))

    r2_word = explained_variance_score(total_true_word, total_est_word, multioutput='raw_values')
    r2_word_adj = np.ones((num_outputs,)) - (np.ones((num_outputs,)) - r2_word) * np.divide(float(num_samples - 1),
                                                                                           float(num_samples - num_feats_all + 3))
    print('Pre adj r2word = {}'.format(r2_word[0]))
    print('Divisor = {}'.format(np.divide(float(num_samples - 1), float(num_samples - num_feats_all + 3))))
    print('Post adj r2word = {}'.format(r2_word_adj[0]))

    time = np.array(np.arange(-1.0, 4.0, 0.002))
    time[np.abs(time) <= 1e-14] = 0.0
    num_time = time.size

    sorted_inds, sorted_reg = sort_sensors()
    r2_all_adj[r2_all_adj < 0.0] = 0.0
    r2_all_adj = np.reshape(r2_all_adj, (306, -1))
    r2_all_adj = r2_all_adj[sorted_inds, :]
    r2_word_adj[r2_word_adj < 0.0] = 0.0
    r2_word_adj = np.reshape(r2_word_adj, (306, -1))
    r2_word_adj = r2_word_adj[sorted_inds, :]

    r2_plot = r2_all_adj - r2_word_adj
    print('Max Unique Variance = {}'.format(np.max(r2_plot)))
    print('Min Unique Variance = {}'.format(np.min(r2_plot)))

    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    # r2_all = np.reshape(r2_all, (306, -1))
    # r2_all = r2_all[sorted_inds, :]
    # fig, ax = plt.subplots()
    # h = ax.imshow(r2_all, interpolation='nearest', aspect='auto', vmin=0.0, vmax=0.6)
    # ax.set_yticks(yticks_sens)
    # ax.set_yticklabels(uni_reg)
    # ax.set_ylabel('Sensors')
    # ax.set_xticks(range(0, num_time, 250))
    # ax.set_xticklabels(time[::250])
    # ax.set_xlabel('Time')
    # ax.set_title('{} {} All\nAdj: {}'.format(exp, sen_type, adj))
    # plt.colorbar(h)


    fig, ax = plt.subplots()
    h = ax.imshow(r2_plot, interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, num_time, 250))
    ax.set_xticklabels(time[::250])
    ax.set_xlabel('Time')
    ax.set_title('{} {} {}\nAdjX: {} AdjY: {} Alg: {} NI: {} avgT: {}'.format(exp, sen_type, word, adjX, adjY, alg, num_instances, avgT))
    plt.colorbar(h)
    plt.savefig('POVE_OH_{}_{}_{}_adjX{}_adjY{}_alg{}_ni{}_avgT{}.pdf'.format(exp, sen_type, word, adjX, adjY, alg, num_instances, avgT), bbox_inches='tight')

    fig, ax = plt.subplots()
    h = ax.imshow(r2_all_adj, interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, num_time, 250))
    ax.set_xticklabels(time[::250])
    ax.set_xlabel('Time')
    ax.set_title('{} {} All\nAdjX: {} AdjY: {} Alg: {} NI: {} avgT: {}'.format(exp, sen_type, adjX, adjY, alg, num_instances, avgT))
    plt.colorbar(h)
    plt.savefig('POVE_OH_{}_{}_all_adjX{}_adjY{}_alg{}_ni{}_avgT{}.pdf'.format(exp, sen_type, adjX, adjY, alg, num_instances, avgT), bbox_inches='tight')

    fig, ax = plt.subplots()
    h = ax.imshow(r2_word_adj, interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, num_time, 250))
    ax.set_xticklabels(time[::250])
    ax.set_xlabel('Time')
    ax.set_title('{} {} less {}\nAdjX: {} AdjY: {} Alg: {} NI: {} avgT: {}'.format(exp, sen_type, word, adjX, adjY, alg, num_instances, avgT))
    plt.colorbar(h)
    plt.savefig('POVE_OH_{}_{}_less-{}_adjX{}_adjY{}_alg{}_ni{}_avgT{}.pdf'.format(exp, sen_type, word, adjX, adjY, alg, num_instances, avgT), bbox_inches='tight')

    plt.show()
