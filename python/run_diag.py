import argparse
import load_data
import models
import numpy as np
import os.path
import random
from sklearn.model_selection import StratifiedKFold

TOP_DIR = '/share/volume0/nrafidi/{exp}_Diag/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}Diag_{sub}_{sen_type}_{word}_o{overlap}_pd{pdtw}_pr{perm}_F{num_folds}_alg{alg}_' \
            'z{zscore}_avg{doAvg}_ni{inst}_nr{rep}_rsPerm{rsP}_rsCV{rsC}_rsSCV{rsS}_{mode}'

CV_RAND_STATE = 12191989
SUB_CV_RAND_STATE = 2282015

VALID_ALGS = ['LASSO', 'ENET', 'GNB']
VALID_SEN_TYPE = ['active', 'passive']
VALID_MODE = ['pred', 'coef']


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
def run_diag_exp(experiment,
                 subject,
                 sen_type,
                 word,
                 overlap,
                 mode='pred',
                 isPDTW = False,
                 isPerm = False,
                 num_folds = 2,
                 alg='GNB',
                 doZscore=False,
                 doAvg=False,
                 num_instances=2,
                 reps_to_use=10,
                 proc=load_data.DEFAULT_PROC,
                 random_state_perm=1,
                 random_state_cv=CV_RAND_STATE,
                 random_state_sub=SUB_CV_RAND_STATE,
                 force=False):
    # Save Directory
    top_dir = TOP_DIR.format(exp=experiment)
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)
    save_dir = SAVE_DIR.format(top_dir=top_dir, sub=subject)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if alg not in VALID_ALGS:
        raise ValueError('invalid alg: must be {}'.format(VALID_ALGS))
    if sen_type not in VALID_SEN_TYPE:
        raise ValueError('invalid sen_type: must be {}'.format(VALID_SEN_TYPE))
    if sen_type not in VALID_MODE:
        raise ValueError('invalid mode: must be {}'.format(VALID_MODE))

    fname = SAVE_FILE.format(dir=save_dir,
                             sub=subject,
                             sen_type=sen_type,
                             word=word,
                             overlap=overlap,
                             pdtw=bool_to_str(isPDTW),
                             perm=bool_to_str(isPerm),
                             num_folds=num_folds,
                             alg=alg,
                             zscore=bool_to_str(doZscore),
                             doAvg=bool_to_str(doAvg),
                             inst=num_instances,
                             rep=reps_to_use,
                             rsP=random_state_perm,
                             rsC=random_state_cv,
                             rsS=random_state_sub,
                             mode=mode)

    if os.path.isfile(fname) and not force:
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
    else:
        data_raw, labels, time = load_data.load_raw(subject=subject,
                                                    word=word,
                                                    sen_type=sen_type,
                                                    experiment=experiment,
                                                    proc=proc)

    data, labels = load_data.avg_data(data_raw=data_raw,
                                      labels_raw=labels,
                                      experiment=experiment,
                                      num_instances=num_instances,
                                      reps_to_use=reps_to_use)
    print(data.shape)

    if isPerm:
        random.seed(random_state_perm)
        random.shuffle(labels)

    tmin = time.min()
    tmax = time.max()

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=CV_RAND_STATE)
    sub_kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SUB_CV_RAND_STATE)

    total_win = int((tmax - tmin) * 500)
    win_starts = range(0, total_win, overlap)

    if total_win > len(time):
        raise ValueError('Windows are messed up.')

    # Run TGM
    if mode == 'pred':
        if alg == 'LASSO':
            (preds, l_ints,
             cv_membership, masks) = models.lasso_tgm(data=data,
                                                      labels=labels,
                                                      kf=kf,
                                                      sub_kf=sub_kf,
                                                      win_starts=win_starts,
                                                      doZscore=doZscore,
                                                      doAvg=doAvg)
        elif alg == 'GNB':
            (preds, l_ints,
             cv_membership, masks) = models.nb_tgm(data=data,
                                                   labels=labels,
                                                   kf=kf,
                                                   sub_kf=sub_kf,
                                                   win_starts=win_starts,
                                                   feature_select='distance_of_means',
                                                   feature_select_params={'number_of_features': num_feats},
                                                   doZscore=doZscore,
                                                   doAvg=doAvg)
        else:
            (preds, l_ints,
             cv_membership, masks) = models.enet_tgm(data=data,
                                                     labels=labels,
                                                     kf=kf,
                                                     sub_kf=sub_kf,
                                                     win_starts=win_starts,
                                                     doZscore=doZscore,
                                                     doAvg=doAvg)

        np.savez_compressed(fname,
                            preds=preds,
                            l_ints=l_ints,
                            cv_membership=cv_membership,
                            masks=masks,
                            time=time,
                            win_starts=win_starts,
                            proc=proc)

    elif mode == 'coef':
        win_lens = load_data.load_win_lens(experiment,
                                           subject,
                                           sen_type,
                                           word,
                                           overlap,
                                           isPDTW,
                                           num_folds,
                                           alg,
                                           doZscore,
                                           doAvg,
                                           num_instances,
                                           reps_to_use,
                                           proc,
                                           random_state_cv,
                                           random_state_sub)
        if alg == 'LASSO':
            coef = models.lasso_tgm_coef(data=data,
                                         labels=labels,
                                         win_starts=win_starts,
                                         win_lens=win_lens,
                                         doZscore=doZscore,
                                         doAvg=doAvg)
        elif alg == 'GNB':
            coef = models.nb_tgm_coef(data=data,
                                      labels=labels,
                                      win_starts=win_starts,
                                      win_len=win_lens,
                                      feature_select='distance_of_means',
                                      feature_select_params={'number_of_features' : num_feats},
                                      doZscore=False,
                                      doAvg=False,
                                      ddof=1)
        else:
            coef = models.enet_tgm_coef(data=data,
                                        labels=labels,
                                        win_starts=win_starts,
                                        win_lens=win_lens,
                                        doZscore=doZscore,
                                        doAvg=doAvg)

        np.savez_compressed(fname,
                            coef=coef,
                            time=time,
                            win_starts=win_starts,
                            proc=proc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--sen_type')
    parser.add_argument('--word')
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--mode', default='pred')
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--isPerm', default='False')
    parser.add_argument('--num_folds', type=int, default=2)
    parser.add_argument('--alg', default='LR')
    parser.add_argument('--doZscore', default='False')
    parser.add_argument('--doAvg', default='False')
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--random_state_perm', type=int, default=1)
    parser.add_argument('--random_state_cv', type=int, default=CV_RAND_STATE)
    parser.add_argument('--random_state_sub', type=int, default=SUB_CV_RAND_STATE)
    parser.add_argument('--force', default='False')

    args = parser.parse_args()

    # Check that parameter setting is valid
    is_valid = args.overlap <= args.win_len
    if not is_valid:
        print('overlap wrong')
    is_valid = is_valid and (args.reps_to_use <= load_data.NUM_REPS[args.experiment])
    if not is_valid:
        print('num reps  wrong')
    is_valid = is_valid and (args.subject in load_data.VALID_SUBS[args.experiment])
    if not is_valid:
        print('subject wrong')
    if args.num_instances != 2:
        is_valid = is_valid and ((args.reps_to_use % args.num_instances) == 0)
    if not is_valid:
        print('instances wrong')
    if args.mode == 'coef':
        is_valid = is_valid and args.num_folds == 2
        if not is_valid:
            print('folds wrong')
    if is_valid:
        run_diag_exp(experiment=args.experiment,
                     subject=args.subject,
                     sen_type=args.sen_type,
                     word=args.word,
                     overlap=args.overlap,
                     mode=args.mode,
                     isPDTW=str_to_bool(args.isPDTW),
                     isPerm=str_to_bool(args.isPerm),
                     num_folds=args.num_folds,
                     alg=args.alg,
                     doZscore=str_to_bool(args.doZscore),
                     doAvg=str_to_bool(args.doAvg),
                     num_instances=args.num_instances,
                     reps_to_use=args.reps_to_use,
                     proc=args.proc,
                     random_state_perm=args.random_state_perm,
                     random_state_cv=args.random_state_cv,
                     random_state_sub=args.random_state_sub,
                     force=str_to_bool(args.force))
    else:
        print('Experiment parameters not valid. Skipping job.')
