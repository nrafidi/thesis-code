import scipy.io
import argparse
import models
import cmumetadata.metaio as cmuio
import numpy as np
import random
import warnings
import string
import os.path
import load_data
from sklearn.model_selection import KFold


# Runs the TGM experiment, 2F CV, separately for active and passive sentences
def run_tgm_exp(experiment,
                subject,
                sen_type,
                word,
                win_len,
                overlap,
                isPDTW = False,
                isPerm = False,
                num_folds = 2,
                alg='LR',
                doZscore=False,
                doAvg=False,
                num_instances=2,
                reps_to_use=10,
                proc=load_data.DEFAULT_PROC,
                random_state=1):

    if isPDTW:
        (time_a, time_p, labels,
         active_data_raw, passive_data_raw) = load_pdtw(subject=subject,
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

    # Run TGM
    (preds, tgm, l_index,
     cv_membership, masks) = models.lr_tgm(data=data,
                                           labels=labels,
                                           kf=kf,
                                           win_starts=win_starts,
                                           win_len=win_len,
                                           doZscore=doZscore,
                                           doAvg=doAvg)

    # Save Directory
    saveDir = '/share/volume0/nrafidi/PA1_TGM'
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    subDir = saveDir + '/' + subject
    if not os.path.exists(subDir):
        os.mkdir(subDir)

    fnameStub = subDir + '/TGM_{sub}_{word}_{aorp}_win{win_len}_zscore{zscore}_avg{doAvg}_2F_LR.mat'
    scipy.io.savemat(
        fnameStub.format(sub=subject, word=word, aorp=actORpass, win_len=win_len, zscore=doZscore, doAvg=doAvg),
        mdict={'preds': preds, 'tgm': tgm, 'feature_masks': masks, 'cv_label': cv_label, 'labels': labels,
               'cv_membership': cv_membership})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--word', required=True)
    parser.add_argument('--win_len', required=True, type=int)
    parser.add_argument('--actORpass', required=False)
    parser.add_argument('--doZscore', required=False, type=int)
    parser.add_argument('--ddof', required=False, type=int)
    parser.add_argument('--doAvg', required=False, type=int)
    parser.add_argument('--do2Samp', required=False, type=int)
    #  parser.add_argument('--doPDTW',required=False,type=int)

    args = parser.parse_args()

    analysis = args.analysis

    {
        'actORpass_krns2_lr': actORpass_krns2_lr,
        'pooled_krns2_lr': pooled_krns2_lr,
    }[analysis](args.experiment, args.subject,
                'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas', args.word,
                args.win_len, args.actORpass, args.doZscore, args.ddof, args.doAvg, args.do2Samp)  # ,args.doPDTW)

# Old slugs:
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'sss_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-5-150_notch-60-120_beatremoval-first_blinkremoval-first'
