import itertools
import os.path
import numpy as np
import batch_experiments_multisub_fold_perm as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'

SAVE_FILE = '{dir}TGM-LOSO_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'


NEW_SAVE_FILE = '{dir}TGM-LOSO_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(['krns2'],
                                   batch_exp.OVERLAPS,
                                   [True],
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TIME_AVGS,
                                   batch_exp.DO_TEST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS)
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
        win_len = grid[8]
        sen = grid[9]
        word = grid[10]

        dir_str = batch_exp.JOB_DIR.format(exp=exp)
        top_dir = TOP_DIR.format(exp=exp)

        perm_list = batch_exp.RANDOM_STATES

        complete_job_perm = SAVE_FILE.format(dir=top_dir,
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
                                        rsP='{}-{}'.format(np.min(perm_list),
                                                           np.max(perm_list)),
                                        mode='acc')

        if not os.path.isfile(complete_job_perm + '.npz'):
            tgm_acc = []
            tgm_pred = []
            cv_membership = []
            for rs in perm_list:
                complete_job = SAVE_FILE.format(dir=top_dir,
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
                                                mode='acc')
                # if os.path.isfile(complete_job + '.npz'):
                #     result = np.load(complete_job + '.npz')
                #     if rs == 0:
                #         l_ints = result['l_ints']
                #         win_starts = result['win_starts']
                #         time = result['time']
                #         proc = result['proc']
                #     cv_membership.append(result['cv_membership'][0])
                #     tgm_acc.append(result['tgm_acc'][None, ...])
                #     tgm_pred.append(result['tgm_pred'][None, ...])
                #     continue
                tgm_acc_perm = []
                tgm_pred_perm = []
                cv_membership_perm = []
                for fold in batch_exp.FOLDS:
                    fname = NEW_SAVE_FILE.format(dir=top_dir,
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
                                             fold=fold)

                    if not os.path.isfile(fname + '.npz'):
                        print('{} missing'.format(fname))
                        break

                    result = np.load(fname + '.npz')
                    if fold == 0:
                        l_ints = result['l_ints']
                        win_starts = result['win_starts']
                        time = result['time']
                        proc = result['proc']
                    cv_membership_perm.append(result['cv_membership'][0])
                    tgm_acc_perm.append(result['tgm_acc'])
                    tgm_pred_perm.append(result['tgm_pred'])

                num_folds = len(batch_exp.FOLDS)
                if exp == 'PassAct3' and word == 'noun2':
                    num_folds /= 2
                if len(tgm_acc_perm) == num_folds:
                    tgm_acc_perm = np.concatenate(tgm_acc_perm, axis=0)
                    tgm_pred_perm = np.concatenate(tgm_pred_perm, axis=0)

                    print(tgm_acc_perm.shape)

                    tgm_acc.append(tgm_acc_perm[None, ...])
                    tgm_pred.append(tgm_pred_perm[None, ...])
                    cv_membership.append(cv_membership_perm)

                    np.savez_compressed(complete_job + '.npz',
                                        l_ints=l_ints,
                                        cv_membership=cv_membership_perm,
                                        tgm_acc=tgm_acc_perm,
                                        tgm_pred=tgm_pred_perm,
                                        win_starts=win_starts,
                                        time=time,
                                        proc=proc)
            if len(tgm_acc) == len(perm_list):
                tgm_acc = np.concatenate(tgm_acc, axis=0)
                tgm_pred = np.concatenate(tgm_pred, axis=0)

                print(tgm_acc.shape)

                np.savez_compressed(complete_job_perm + '.npz',
                                    l_ints=l_ints,
                                    cv_membership=cv_membership,
                                    tgm_acc=tgm_acc,
                                    tgm_pred=tgm_pred,
                                    win_starts=win_starts,
                                    time=time,
                                    proc=proc)