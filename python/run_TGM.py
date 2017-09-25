import argparse
import load_data
import models
import numpy as np
import os.path
import random
from sklearn.model_selection import KFold

SAVE_DIR = '/share/volume0/nrafidi/{exp}_TGM/{sub}/'
SAVE_FILE = '{dir}TGM_{sub}_{sen_type}_{word}_w{win_len}_o{overlap}_pd{pdtw}_pr{perm}_{num_folds}F_{alg}_' \
            'z{zscore}_avg{doAvg}_ni{inst}_nr{rep}_rs{rs}_{mode}'

CV_RAND_STATE = 12191989

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
def run_tgm_exp(experiment,
                subject,
                sen_type,
                word,
                win_len,
                overlap,
                mode='pred',
                isPDTW = False,
                isPerm = False,
                num_folds = 2,
                alg='LR',
                num_feats = 500,
                doZscore=False,
                doAvg=False,
                num_instances=2,
                reps_to_use=10,
                proc=load_data.DEFAULT_PROC,
                random_state=1,
                force=False):
    # Save Directory
    saveDir = SAVE_DIR.format(exp=experiment, sub=subject)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    if alg == 'LR':
        alg_str = alg
    elif alg == 'GNB':
        alg_str = alg + '-{}'.format(num_feats)
    else:
        raise ValueError('invalid alg: must be LR or GNB')

    fname = SAVE_FILE.format(dir=saveDir,
                             sub=subject,
                             sen_type=sen_type,
                             word=word,
                             win_len=win_len,
                             overlap=overlap,
                             pdtw=bool_to_str(isPDTW),
                             perm=bool_to_str(isPerm),
                             num_folds=num_folds,
                             alg=alg_str,
                             zscore=bool_to_str(doZscore),
                             doAvg=bool_to_str(doAvg),
                             inst=num_instances,
                             rep=reps_to_use,
                             rs=random_state,
                             mode=mode)

    print(fname)

    if os.path.isfile(fname) and not force:
        print('Job already completed. Skipping Job.')
        return

    random.seed(random_state)

    if isPDTW:
        (time_a, time_p, labels,
         active_data_raw, passive_data_raw) = load_data.load_pdtw(subject=subject,
                                                                  word=word,
                                                                  experiment=experiment,
                                                                  proc=proc)
        if sen_type == 'active':
            data_raw = active_data_raw
            time = time_a
        elif sen_type == 'passive':
            data_raw = passive_data_raw
            time = time_p
        else:
            raise ValueError('invalid sen_type: must be active or passive')
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
        random.shuffle(labels)

    tmin = time.min()
    tmax = time.max()

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=CV_RAND_STATE)

    total_win = int((tmax - tmin) * 500)
    win_starts = range(0, total_win - win_len, overlap)

    assert total_win <= len(time)

    # Run TGM
    if alg == 'LR':
        if mode == 'pred':
            (preds, l_ints,
             cv_membership, masks) = models.lr_tgm(data=data,
                                                   labels=labels,
                                                   kf=kf,
                                                   win_starts=win_starts,
                                                   win_len=win_len,
                                                   doZscore=doZscore,
                                                   doAvg=doAvg)
        elif mode == 'coef':
            coef = models.lr_tgm_coef(data=data,
                                      labels=labels,
                                      win_starts=win_starts,
                                      win_len=win_len,
                                      doZscore=doZscore,
                                      doAvg=doAvg)
        else:
            raise ValueError('invalid mode: must be pred or coef')
    elif alg == 'GNB':

        if mode == 'pred':
            (preds, l_ints,
             cv_membership, masks) = models.nb_tgm(data=data,
                                                   labels=labels,
                                                   kf=kf,
                                                   win_starts=win_starts,
                                                   win_len=win_len,
                                                   feature_select='distance_of_means',
                                                   feature_select_params={'number_of_features' : num_feats},
                                                   doZscore=doZscore,
                                                   doAvg=doAvg)
        elif mode == 'coef':
            coef = models.nb_tgm_coef(data=data,
                                      labels=labels,
                                      win_starts=win_starts,
                                      win_len=win_len,
                                      feature_select='distance_of_means',
                                      feature_select_params={'number_of_features' : num_feats},
                                      doZscore=False,
                                      doAvg=False,
                                      ddof=1)
        else:
            raise ValueError('invalid mode: must be pred or coef')
    else:
        raise ValueError('invalid alg: must be LR or GNB')

    if mode == 'pred':
        np.savez_compressed(fname,
                            preds=preds,
                            l_ints=l_ints,
                            cv_membership=cv_membership,
                            masks=masks,
                            time=time,
                            win_starts=win_starts,
                            proc=proc)
    elif mode == 'coef':
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
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--mode', default='pred')
    parser.add_argument('--isPDTW', default='False')
    parser.add_argument('--isPerm', default='False')
    parser.add_argument('--num_folds', type=int, default=2)
    parser.add_argument('--alg', default='LR')
    parser.add_argument('--num_feats', type=int, default=50)
    parser.add_argument('--doZscore', default='False')
    parser.add_argument('--doAvg', default='False')
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--force', default='False')

    args = parser.parse_args()

    # Check that parameter setting is valid
    is_valid = args.overlap <= args.win_len
    is_valid = is_valid and (args.reps_to_use <= load_data.NUM_REPS[args.experiment])
    is_valid = is_valid and (args.subject in load_data.VALID_SUBS[args.experiment])
    if args.num_instances != 2:
        is_valid = is_valid and ((args.reps_to_use % args.num_instances) == 0)
    if is_valid:
        run_tgm_exp(experiment=args.experiment,
                    subject=args.subject,
                    sen_type=args.sen_type,
                    word=args.word,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    mode=args.mode,
                    isPDTW=str_to_bool(args.isPDTW),
                    isPerm=str_to_bool(args.isPerm),
                    num_folds=args.num_folds,
                    alg=args.alg,
                    num_feats=args.num_feats,
                    doZscore=str_to_bool(args.doZscore),
                    doAvg=str_to_bool(args.doAvg),
                    num_instances=args.num_instances,
                    reps_to_use=args.reps_to_use,
                    proc=args.proc,
                    random_state=args.random_state,
                    force=str_to_bool(args.force))
    else:
        print('Experiment parameters not valid. Skipping job.')
