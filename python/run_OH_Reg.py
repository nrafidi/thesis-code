import argparse
import load_data_ordered as load_data
import models
import numpy as np
import os.path
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_OH/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}OH_{sub}_{sen_type}_{word}_pr{perm}_' \
            'F{num_folds}_alg{alg}_adj{adj}_ni{inst}_nr{rep}_rsPerm{rsP}_rsCV{rsC}'

CV_RAND_STATE = 12191989

VALID_ALGS = ['ols', 'ridge', 'lasso', 'enet']
VALID_SEN_TYPE = ['active', 'passive']

WORD_COLS = {'krns2': {'art1': 0,
                       'noun1': 1,
                       'verb': 2,
                       'art2': 3,
                       'noun2': 4},
             'PassAct2': {'noun1': 0,
                          'verb': 1,
                          'noun2': 2},
             'PassAct3': {'noun1': 0,
                          'verb': 1,
                          'noun2': 2}
             }


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'

def load_one_hot(labels):
    unique_labels = np.unique(labels)
    num_labels = unique_labels.size
    one_hot_dict = {}
    for i_l, uni_l in enumerate(unique_labels):
        one_hot_dict[uni_l] = np.zeros((num_labels,))
        one_hot_dict[uni_l][i_l] = 1
    one_hot = []
    for l in labels:
        one_hot.append(one_hot_dict[l])
    return np.stack(one_hot)


# Runs the TGM experiment
def run_sv_exp(experiment,
               subject,
               sen_type,
               word,
               isPerm = False,
               num_folds = 16,
               alg='ols',
               adj='mean_center',
               num_instances=1,
               reps_to_use=10,
               proc=load_data.DEFAULT_PROC,
               random_state_perm=1,
               random_state_cv=CV_RAND_STATE,
               force=False):
    # warnings.filterwarnings('error')
    # Save Directory
    top_dir = TOP_DIR.format(exp=experiment)
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)
    save_dir = SAVE_DIR.format(top_dir=top_dir, sub=subject)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname = SAVE_FILE.format(dir=save_dir,
                             sub=subject,
                             sen_type=sen_type,
                             word=word,
                             perm=bool_to_str(isPerm),
                             num_folds=num_folds,
                             alg=alg,
                             adj=adj,
                             inst=num_instances,
                             rep=reps_to_use,
                             rsP=random_state_perm,
                             rsC=random_state_cv)


    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return

    data, labels, time, final_inds = load_data.load_sentence_data(subject=subject,
                                                                  word='noun1',
                                                                  sen_type=sen_type,
                                                                  experiment=experiment,
                                                                  proc=proc,
                                                                  num_instances=num_instances,
                                                                  reps_to_use=reps_to_use,
                                                                  noMag=False,
                                                                  sorted_inds=None)

    l_set = np.unique(labels)
    n_l = len(l_set)
    l_index = {l_set[i]: i for i in xrange(n_l)}
    l_ints = np.array([l_index[l] for l in labels])

    semantic_vectors = []
    for col in range(labels.shape[-1]):
        if word != 'all':
            if col == WORD_COLS[experiment][word]:
                continue
        oh = load_one_hot(labels[:, col])
        semantic_vectors.append(oh)

    semantic_vectors = np.stack(semantic_vectors, axis=1)
    print(semantic_vectors)

    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)

    if num_folds > 8:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state_cv)
    else:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state_cv)

    preds, l_ints, cv_membership, scores, test_data_all = models.lin_reg(data,
                                                                         semantic_vectors,
                                                                         l_ints,
                                                                         kf,
                                                                         reg=alg,
                                                                         adj=adj,
                                                                         ddof=1)
    np.savez_compressed(fname,
                        preds=preds,
                        l_ints=l_ints,
                        cv_membership=cv_membership,
                        scores=scores,
                        test_data_all=test_data_all,
                        time=time,
                        proc=proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--sen_type', choices=VALID_SEN_TYPE)
    parser.add_argument('--word', default='all')
    parser.add_argument('--isPerm', action='store_true')
    parser.add_argument('--num_folds', type=int, default=16)
    parser.add_argument('--alg', default='ols', choices=VALID_ALGS)
    parser.add_argument('--adj', default='mean_center')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--cv_random_state', type=int, default=CV_RAND_STATE)
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    # Check that parameter setting is valid
    total_valid = True
    is_valid = args.num_folds <= 16*args.num_instances
    total_valid = total_valid and is_valid
    if not is_valid:
        print('num folds wrong {} {}'.format(args.num_folds, args.num_instances))
    is_valid = args.reps_to_use <= load_data.NUM_REPS[args.experiment]
    total_valid = total_valid and is_valid
    if not is_valid:
        print('num reps  wrong')
    is_valid = args.subject in load_data.VALID_SUBS[args.experiment]
    total_valid = total_valid and is_valid
    if not is_valid:
        print('subject wrong')
    if args.num_instances != 2:
        is_valid = (args.reps_to_use % args.num_instances) == 0
        total_valid = total_valid and is_valid
        if not is_valid:
            print('instances wrong')
    if total_valid:
        run_sv_exp(experiment=args.experiment,
                   subject=args.subject,
                   sen_type=args.sen_type,
                   word=args.word,
                   isPerm=args.isPerm,
                   num_folds=args.num_folds,
                   alg=args.alg,
                   adj=args.adj,
                   num_instances=args.num_instances,
                   reps_to_use=args.reps_to_use,
                   proc=args.proc,
                   random_state_perm=args.perm_random_state,
                   random_state_cv=args.cv_random_state,
                   force=args.force)
    else:
        print('Experiment parameters not valid. Skipping job.')
