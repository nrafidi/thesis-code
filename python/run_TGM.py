import argparse
import load_data
import models
import numpy as np
import os.path
from random import shuffle
import sklearn.model_selection.KFold as KFold

SAVE_DIR = '/share/volume0/nrafidi/{exp}_TGM/{sub}'
SAVE_FILE = '{dir}/TGM_{sub}_{sen_type}_{word}_w{win_len}_o{overlap}_pd{pdtw}_pr{perm}_{num_folds}F_{alg}_z{zscore}_avg{doAvg}_ni{inst}_nr{rep}_rs{rs}_{mode}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


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
                random_state=1):

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
    tmin = time.min()
    tmax = time.max()

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    total_win = int((tmax - tmin) * 500)
    win_starts = range(0, total_win - win_len, overlap)

    assert total_win <= len(time)

    # Run TGM
    if alg == 'LR':
        alg_str = alg
        if mode == 'pred':
            (preds, l_index,
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
        alg_str = alg + '-{}'.format(num_feats)

        if mode == 'pred':
            (preds, l_index,
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

    # Save Directory
    saveDir = SAVE_DIR.format(exp=experiment, sub=subject)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

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

    if mode == 'pred':
        np.savez_compressed(fname,
                            preds=preds,
                            l_index=l_index,
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
    parser.add_argument('--mode')
    parser.add_argument('--isPDTW', type=bool, default=False)
    parser.add_argument('--isPerm', type=bool, default=False)
    parser.add_argument('--num_folds', type=int, default=2)
    parser.add_argument('--alg', default='LR')
    parser.add_argument('--num_feats', type=int, default=500)
    parser.add_argument('--doZscore', type=bool, default=False)
    parser.add_argument('--doAvg', type=bool, default=False)
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--reps_to_use', type=int, default=10)
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
    parser.add_argument('--random_state', type=int, default=1)

    args = parser.parse_args()

    run_tgm_exp(experiment=args.experiment,
                subject=args.subject,
                sen_type=args.sen_type,
                word=args.word,
                win_len=args.win_len,
                overlap=args.overlap,
                mode=args.mode,
                isPDTW=args.isPDTW,
                isPerm=args.isPerm,
                num_folds=args.num_folds,
                alg=args.alg,
                num_feats=args.num_feats,
                doZscore=args.doZscore,
                doAvg=args.doAvg,
                num_instances=args.num_instances,
                reps_to_use=args.reps_to_use,
                proc=args.proc,
                random_state=args.random_state)