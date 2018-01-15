import argparse
import itertools
import matplotlib
matplotlib.use('TkAgg')
import load_data_ordered as load_data
import matplotlib.pyplot as plt
import numpy as np
import run_OH_Reg
import scipy.io as sio
from sklearn.metrics import explained_variance_score
import os.path

NUM_FEATS = {'krns2': 16,
             'PassAct2': 12,
             'PassAct3': 12}

LOO = {1: 16,
       2: 32,
       5: 80,
       10: 160}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['krns2', 'PassAct2', 'PassAct3'], default='krns2')
    parser.add_argument('--sen_type', choices=['active', 'passive'], default='active')
    parser.add_argument('--word', choices=['noun1', 'verb', 'noun2'], default='noun1')
    parser.add_argument('--adj', choices=[None, 'mean_center', 'zscore'], default=None)
    parser.add_argument('--alg', choices=['ols', 'ridge'], default='ols')
    args = parser.parse_args()

    exp = args.experiment
    subjects = load_data.VALID_SUBS[exp]
    num_feats_all = NUM_FEATS[exp]
    sen_type = args.sen_type
    word = args.word
    adj = args.adj
    alg = args.alg

    top_dir = run_OH_Reg.TOP_DIR.format(exp=exp)

    num_instances_to_try = [1, 2, 10]
    max_uni_var = []
    num_uni_var = []

    for num_instances in num_instances_to_try:
        num_folds = LOO[num_instances]

        total_est_all = []
        total_true_all = []
        total_est_word = []
        total_true_word = []
        for subject in subjects:
            save_dir = run_OH_Reg.SAVE_DIR.format(top_dir=top_dir, sub=subject)
            fname = run_OH_Reg.SAVE_FILE.format(dir=save_dir,
                                     sub=subject,
                                     sen_type=sen_type,
                                     word='all',
                                     perm='F',
                                     num_folds=num_folds,
                                     alg=alg,
                                     adj=adj,
                                     inst=num_instances,
                                     rep=10,
                                     rsP=1,
                                     rsC=run_OH_Reg.CV_RAND_STATE)
            if os.path.isfile(fname + '.npz'):
                result = np.load(fname + '.npz')
                total_est_all.append(result['preds'])
                total_true_all.append(result['test_data_all'])

            fname = run_OH_Reg.SAVE_FILE.format(dir=save_dir,
                                                sub=subject,
                                                sen_type=sen_type,
                                                word=word,
                                                perm='F',
                                                num_folds=num_folds,
                                                alg=alg,
                                                adj=adj,
                                                inst=num_instances,
                                                rep=10,
                                                rsP=1,
                                                rsC=run_OH_Reg.CV_RAND_STATE)
            if os.path.isfile(fname + '.npz'):
                result = np.load(fname + '.npz')
                total_est_word.append(result['preds'])
                total_true_word.append(result['test_data_all'])

        total_est_all = np.concatenate(total_est_all, axis=0)
        total_true_all = np.concatenate(total_true_all, axis=0)
        total_est_word = np.concatenate(total_est_word, axis=0)
        total_true_word = np.concatenate(total_true_word, axis=0)

        num_samples = total_est_all.shape[0]
        num_outputs = total_est_all.shape[-1]

        r2_all = explained_variance_score(total_true_all, total_est_all, multioutput='raw_values')
        r2_all_adj = np.ones((num_outputs,)) - (np.ones((num_outputs,)) - r2_all)*np.divide(float(num_samples - 1),
                                                                                            float(num_samples-num_feats_all-1))

        r2_word = explained_variance_score(total_true_word, total_est_word, multioutput='raw_values')
        r2_word_adj = np.ones((num_outputs,)) - (np.ones((num_outputs,)) - r2_word) * np.divide(float(num_samples - 1),
                                                                                               float(num_samples - num_feats_all + 3))

        r2_all_adj[r2_all_adj < 0.0] = 0.0
        r2_word_adj[r2_word_adj < 0.0] = 0.0

        r2_plot = r2_all_adj - r2_word_adj
        max_uni_var.append(np.max(r2_plot))
        num_uni_var.append(np.sum(r2_plot > 0.0))

    fig, ax = plt.subplots()
    ax.plot(num_instances_to_try, max_uni_var)
    ax.set_ylabel('Maximum Unique Variance')
    ax.set_xticks(num_instances_to_try)
    ax.set_xlabel('Number of instances per sentence')
    ax.set_title(
        '{} {} {}\nAdj: {} Alg: {}'.format(exp, sen_type, word, adj, alg))
    plt.savefig('POVE_OH_{}_{}_{}_adj{}_alg{}_max.png'.format(exp, sen_type, word, adj, alg), bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(num_instances_to_try, num_uni_var)
    ax.set_ylabel('Number of Points with Unique Variance > 0.0')
    ax.set_xticks(num_instances_to_try)
    ax.set_xlabel('Number of instances per sentence')
    ax.set_title(
        '{} {} {}\nAdj: {} Alg: {}'.format(exp, sen_type, word, adj, alg))
    plt.savefig('POVE_OH_{}_{}_{}_adj{}_alg{}_num.png'.format(exp, sen_type, word, adj, alg), bbox_inches='tight')