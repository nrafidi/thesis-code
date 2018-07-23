import itertools
import os.path
import batch_experiments_eos_kf_multisub_fold as batch_exp
import numpy as np

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_KF_EOS/'
SAVE_FILE = '{dir}TGM-LOSO-{k}F_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_rsCV{rsCV}_{fold}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.DO_TST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.CV_RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.NUM_FOLDS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        exp = grid[0]
        overlap = grid[1]
        isPerm = grid[2]
        alg = grid[3]
        adj = grid[4]
        tm_avg = grid[5]
        tst_avg = grid[6]
        ni = grid[7]
        rs = grid[8]
        cv_rs = grid[9]
        win_len = grid[10]
        sen = grid[11]
        word = grid[12]
        num_folds = grid[13]

        if exp == 'krns2' and word == 'senlen':
            continue

        if word in ['propid', 'voice', 'senlen', 'noun1'] and sen != 'pooled':
            continue

        if num_folds > 2 and word == 'propid':
            continue


        top_dir = TOP_DIR.format(exp=exp)

        complete_job = fname = SAVE_FILE.format(dir=top_dir,
                                     sen_type=sen,
                                     word=word,
                                     win_len=win_len,
                                     ov=overlap,
                                     perm=bool_to_str(isPerm),
                                     alg=alg,
                                     adj=adj,
                                     avgTm=bool_to_str(tm_avg),
                                     avgTst=bool_to_str(tst_avg),
                                     inst=ni,
                                     rsP=rs,
                                     rsCV=cv_rs,
                                     k=num_folds,
                                     fold='acc')

        if os.path.isfile(complete_job + '.npz'):
            continue
        tgm_acc = []
        tgm_pred = []
        cv_membership = []
        for fold in range(num_folds):
            fname = SAVE_FILE.format(dir=top_dir,
                                     sen_type=sen,
                                     word=word,
                                     win_len=win_len,
                                     ov=overlap,
                                     perm=bool_to_str(isPerm),
                                     alg=alg,
                                     adj=adj,
                                     avgTm=bool_to_str(tm_avg),
                                     avgTst=bool_to_str(tst_avg),
                                     inst=ni,
                                     rsP=rs,
                                     rsCV=cv_rs,
                                     k=num_folds,
                                     fold=fold)

            if not os.path.isfile(fname + '.npz'):
                print('{} is missing'.format(fname))
                break

            result = np.load(fname + '.npz')
            if fold == 0:
                l_ints = result['l_ints']
                win_starts = result['win_starts']

                time = result['time']
                proc = result['proc']
            cv_membership.append(result['cv_membership'][0])
            tgm_acc.append(result['tgm_acc'])
            tgm_pred.append(result['tgm_pred'])

        if len(tgm_acc) == num_folds:
            tgm_acc = np.concatenate(tgm_acc, axis=0)
            tgm_pred = np.concatenate(tgm_pred, axis=0)

            print(tgm_acc.shape)

            np.savez_compressed(complete_job + '.npz',
                                l_ints=l_ints,
                                cv_membership=cv_membership,
                                tgm_acc=tgm_acc,
                                tgm_pred=tgm_pred,
                                win_starts=win_starts,
                                time=time,
                                proc=proc)