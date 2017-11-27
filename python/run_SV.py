import argparse
import load_data
import models
import numpy as np
import os.path
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import warnings

TOP_DIR = '/share/volume0/nrafidi/{exp}_SV/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}SV_{sub}_{sen_type}_{word}_{model}_art1{art1}_art2{art2}_{direction}_pca{pca}_pd{pdtw}_pr{perm}_' \
            'F{num_folds}_alg{alg}_adj{adj}_ni{inst}_nr{rep}_rsPerm{rsP}_rsCV{rsC}_rsSCV{rsS}'

CV_RAND_STATE = 12191989
SUB_CV_RAND_STATE = 2282015

VALID_ALGS = ['ridge', 'lasso', 'enet']
VALID_SEN_TYPE = ['active', 'passive']


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


def str_to_bool(str_bool):
    if str_bool == 'False':
        return False
    else:
        return True


# Runs the TGM experiment
def run_sv_exp(experiment,
               subject,
               sen_type,
               word,
               model='one_hot',
               inc_art1=False,
               inc_art2=False,
               only_art1=False,
               only_art2=False,
               direction='encoding',
               doPCA = False,
               isPDTW = False,
               isPerm = False,
               num_folds = 2,
               alg='ridge',
               adj='mean_center',
               num_instances=1,
               reps_to_use=10,
               proc=load_data.DEFAULT_PROC,
               random_state_perm=1,
               random_state_cv=CV_RAND_STATE,
               random_state_sub=SUB_CV_RAND_STATE,
               force=False):
    # warnings.filterwarnings('error')
    # Save Directory
    top_dir = TOP_DIR.format(exp=experiment)
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)
    save_dir = SAVE_DIR.format(top_dir=top_dir, sub=subject)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if alg not in VALID_ALGS:
        raise ValueError('invalid alg {}: must be {}'.format(alg, VALID_ALGS))
    if sen_type not in VALID_SEN_TYPE:
        raise ValueError('invalid sen_type {}: must be {}'.format(sen_type, VALID_SEN_TYPE))

    if only_art1:
        art1_str = 'O'
        art2_str = 'F'
    elif only_art2:
        art2_str = 'O'
        art1_str = 'F'
    else:
        art1_str = bool_to_str(inc_art1)
        art2_str = bool_to_str(inc_art2)

    fname = SAVE_FILE.format(dir=save_dir,
                             sub=subject,
                             sen_type=sen_type,
                             word=word,
                             model=model,
                             art1=art1_str,
                             art2=art2_str,
                             direction=direction,
                             pca=bool_to_str(doPCA),
                             pdtw=bool_to_str(isPDTW),
                             perm=bool_to_str(isPerm),
                             num_folds=num_folds,
                             alg=alg,
                             adj=adj,
                             inst=num_instances,
                             rep=reps_to_use,
                             rsP=random_state_perm,
                             rsC=random_state_cv,
                             rsS=random_state_sub)


    if os.path.isfile(fname + '.npz') and not force:
        print('Job already completed. Skipping Job.')
        print(fname)
        return


    if isPDTW:
        (time_a, time_p, labels,
         active_data_raw, passive_data_raw) = load_data.load_pdtw(subject=subject,
                                                                  word=word,
                                                                  experiment=experiment,
                                                                  proc=proc)
        if sen_type == 'active':
            data_raw = active_data_raw
            time = time_a
        else:
            data_raw = passive_data_raw
            time = time_p
        sentence_ids = range(data_raw.shape[0])
    else:
        data_raw, labels, time, sentence_ids = load_data.load_raw(subject=subject,
                                                       word=word,
                                                       sen_type=sen_type,
                                                       experiment=experiment,
                                                       proc=proc)

    print(data_raw.shape)

    data, labels, sentence_ids = load_data.avg_data(data_raw=data_raw,
                                         labels_raw=labels,
                                         sentence_ids_raw=sentence_ids,
                                         experiment=experiment,
                                         num_instances=num_instances,
                                         reps_to_use=reps_to_use)
    print(data.shape)

    l_set = np.unique(labels)
    n_l = len(l_set)
    l_index = {l_set[i]: i for i in xrange(n_l)}
    l_ints = np.array([l_index[l] for l in labels])

    if model == 'glove':
        semantic_vectors = load_data.load_glove_vectors(labels)
    else:
        if only_art1:
            semantic_vectors = load_data.load_one_hot(load_data.get_arts_from_senid(sentence_ids, 1))
        elif only_art2:
            semantic_vectors = load_data.load_one_hot(load_data.get_arts_from_senid(sentence_ids, 2))
        else:
            semantic_vectors = load_data.load_one_hot(labels)
            if inc_art1:
                semantic_vectors = np.concatenate([semantic_vectors,
                                                   load_data.load_one_hot(load_data.get_arts_from_senid(sentence_ids, 1))],
                                                  axis=1)
            if inc_art2:
                semantic_vectors = np.concatenate([semantic_vectors,
                                                   load_data.load_one_hot(
                                                       load_data.get_arts_from_senid(sentence_ids, 2))],
                                                  axis=1)

    if doPCA:
        if direction == 'encoding':
            pca = PCA()
            pca.fit(semantic_vectors)
            semantic_vectors *= np.transpose(pca.components_)
        else:
            reshaped_data = np.reshape(data, (data.shape[0], -1))
            pca = PCA()
            pca.fit(semantic_vectors)
            semantic_vectors *= np.transpose(pca.components_)



    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)

    if num_folds > 8:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state_cv)
    else:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state_cv)

    preds, l_ints, cv_membership, scores = models.lin_reg(data,
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
                        time=time,
                        proc=proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--sen_type')
    parser.add_argument('--word')
    parser.add_argument('--model', default='one_hot')
    parser.add_argument('--inc_art1', default='False')
    parser.add_argument('--inc_art2', default='False')
    parser.add_argument('--only_art1', default='False')
    parser.add_argument('--only_art2', default='False')
    parser.add_argument('--direction', default='encoding')
    parser.add_argument('--doPCA', default='False')
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--isPerm', default='False')
    parser.add_argument('--num_folds', type=int, default=16)
    parser.add_argument('--alg', default='ridge')
    parser.add_argument('--adj', default='mean_centre')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--perm_random_state', type=int, default=1)
    parser.add_argument('--cv_random_state', type=int, default=CV_RAND_STATE)
    parser.add_argument('--sub_random_state', type=int, default=SUB_CV_RAND_STATE)
    parser.add_argument('--force', default='False')

    args = parser.parse_args()

    # Check that parameter setting is valid
    total_valid = True

    inc_art1 = str_to_bool(args.inc_art1)
    inc_art2 = str_to_bool(args.inc_art2)
    only_art1 = str_to_bool(args.only_art1)
    only_art2 = str_to_bool(args.only_art2)

    if inc_art1 or inc_art2 or only_art2 or only_art1:
        is_valid = args.model == 'one_hot'
        total_valid = total_valid and is_valid
        if not is_valid:
            print('wrong model for articles')
    is_valid = not ((only_art1 and only_art2) or (only_art1 and inc_art2) or (only_art2 and inc_art1))
    total_valid = total_valid and is_valid
    if not is_valid:
        print('articles wrong')
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
                   model=args.model,
                   inc_art1=inc_art1,
                   inc_art2=inc_art2,
                   only_art1=only_art1,
                   only_art2=only_art2,
                   direction=args.direction,
                   doPCA=str_to_bool(args.doPCA),
                   isPDTW=str_to_bool(args.isPDTW),
                   isPerm=str_to_bool(args.isPerm),
                   num_folds=args.num_folds,
                   alg=args.alg,
                   adj=args.adj,
                   num_instances=args.num_instances,
                   reps_to_use=args.reps_to_use,
                   proc=args.proc,
                   random_state_perm=args.perm_random_state,
                   random_state_cv=args.cv_random_state,
                   random_state_sub=args.sub_random_state,
                   force=str_to_bool(args.force))
    else:
        print('Experiment parameters not valid. Skipping job.')
