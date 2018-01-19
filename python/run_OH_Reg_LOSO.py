import argparse
import load_data_ordered as load_data
import models
import numpy as np
import os.path
import random
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_OH/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}OH-LOSO_{sub}_{sen_type}_{word}_pr{perm}_' \
            'alg{alg}_adjX-{adjX}_adjY-{adjY}_avgTest{avgT}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}'

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
               alg='ols',
               adjX='mean_center',
               adjY='mean_center',
               doTestAvg=True,
               num_instances=1,
               reps_to_use=10,
               proc=load_data.DEFAULT_PROC,
               random_state_perm=1,
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
                             alg=alg,
                             adjX=adjX,
                             adjY=adjY,
                             avgT=bool_to_str(doTestAvg),
                             inst=num_instances,
                             rep=reps_to_use,
                             rsP=random_state_perm)


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

    l_set = np.unique(labels, axis=0).tolist()
    num_labels = labels.shape[0]
    l_ints = np.empty((num_labels,))
    for i_l in range(num_labels):
        for j_l, l in enumerate(l_set):
            if np.all(l == labels[i_l, :]):
                l_ints[i_l] = j_l
                break
    print(l_ints)

    semantic_vectors = []
    for col in range(labels.shape[-1]):
        if word != 'all':
            if col == WORD_COLS[experiment][word]:
                continue
        oh = load_one_hot(labels[:, col])
        print(oh.shape)
        semantic_vectors.append(oh)

    semantic_vectors = np.concatenate(semantic_vectors, axis=1)
    print(semantic_vectors)

    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)

    preds, l_ints, cv_membership, scores, test_data_all, weights, bias = models.lin_reg_loso(data,
                                                                                             semantic_vectors,
                                                                                             l_ints,
                                                                                             reg=alg,
                                                                                             adjX=adjX,
                                                                                             adjY=adjY,
                                                                                             doTestAvg=doTestAvg,
                                                                                             ddof=1)
    np.savez_compressed(fname,
                        preds=preds,
                        l_ints=l_ints,
                        cv_membership=cv_membership,
                        scores=scores,
                        test_data_all=test_data_all,
                        weights=weights,
                        bias=bias,
                        time=time,
                        proc=proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--sen_type', choices=VALID_SEN_TYPE)
    parser.add_argument('--word', default='all')
    parser.add_argument('--isPerm', action='store_true')
    parser.add_argument('--alg', default='ols', choices=VALID_ALGS)
    parser.add_argument('--adjX', default='mean_center')
    parser.add_argument('--adjY', default='mean_center')
    parser.add_argument('--doTestAvg', action='store_true')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    # Check that parameter setting is valid
    total_valid = True
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
                   alg=args.alg,
                   adjX=args.adjX,
                   adjY=args.adjY,
                   doTestAvg=args.doTestAvg,
                   num_instances=args.num_instances,
                   reps_to_use=args.reps_to_use,
                   proc=args.proc,
                   random_state_perm=args.perm_random_state,
                   force=args.force)
    else:
        print('Experiment parameters not valid. Skipping job.')
