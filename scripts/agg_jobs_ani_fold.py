import itertools
import os.path
import numpy as np
import batch_experiments_ani_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO-ANI_{sub}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'

NEW_SAVE_FILE = '{dir}TGM-LOSO-ANI_{sub}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.FOLDS,
                                   batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TIME_AVGS,
                                   batch_exp.DO_TEST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SUBJECTS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:

        fold = grid[0]
        exp = grid[1]
        overlap = grid[2]
        isPerm = grid[3]
        alg = grid[4]
        adj = grid[5]
        tm_avg = grid[6]
        tst_avg = grid[7]
        ni = grid[8]
        rs = grid[9]
        win_len = grid[10]
        sub = grid[11]

        dir_str = batch_exp.JOB_DIR.format(exp=exp)
        top_dir = TOP_DIR.format(exp=exp)
        save_dir = SAVE_DIR.format(top_dir=top_dir, sub=sub)
        complete_job = SAVE_FILE.format(dir=save_dir,
                                        sub=sub,
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
        if os.path.isfile(complete_job + '.npz'):
            continue
        tgm_acc_n1n1 = []
        tgm_pred_n1n1 = []
        tgm_acc_n1n2 = []
        tgm_pred_n1n2 = []
        tgm_acc_n2n1 = []
        tgm_pred_n2n1 = []
        tgm_acc_n2n2 = []
        tgm_pred_n2n2 = []
        for fold in batch_exp.FOLDS:
            fname = NEW_SAVE_FILE.format(dir=save_dir,
                                     sub=sub,
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
                n1_win_starts = result['n1_win_starts']
                n2_win_starts = result['n2_win_starts']
                cv_membership = result['cv_membership']
                n1_time = result['n1_time']
                n2_time = result['n2_time']
                proc = result['proc']
            tgm_acc = result['tgm_acc']
            tgm_pred = result['tgm_pred']
            tgm_acc_n1n1.append(tgm_acc[0, 0])
            tgm_acc_n1n2.append(tgm_acc[0, 1])
            tgm_acc_n2n1.append(tgm_acc[1, 0])
            tgm_acc_n2n2.append(tgm_acc[1, 1])
            tgm_pred_n1n1.append(tgm_pred[0, 0])
            tgm_pred_n1n2.append(tgm_pred[0, 1])
            tgm_pred_n2n1.append(tgm_pred[1, 0])
            tgm_pred_n2n2.append(tgm_pred[1, 1])

        if len(tgm_acc_n1n1) == len(batch_exp.FOLDS):
            tgm_acc_n1n1 = np.concatenate(tgm_acc_n1n1, axis=0)
            tgm_acc_n1n2 = np.concatenate(tgm_acc_n1n2, axis=0)
            tgm_acc_n2n1 = np.concatenate(tgm_acc_n2n1, axis=0)
            tgm_acc_n2n2 = np.concatenate(tgm_acc_n2n2, axis=0)
            tgm_pred_n1n1 = np.concatenate(tgm_pred_n1n1, axis=0)
            tgm_pred_n1n2 = np.concatenate(tgm_pred_n1n2, axis=0)
            tgm_pred_n2n1 = np.concatenate(tgm_pred_n2n1, axis=0)
            tgm_pred_n2n2 = np.concatenate(tgm_pred_n2n2, axis=0)
            

            print(tgm_acc.shape)

            np.savez_compressed(complete_job + '.npz',
                                l_ints=l_ints,
                                cv_membership=cv_membership,
                                tgm_acc_n1n1=tgm_acc_n1n1,
                                tgm_acc_n1n2=tgm_acc_n1n2,
                                tgm_acc_n2n1=tgm_acc_n2n1,
                                tgm_acc_n2n2=tgm_acc_n2n2,
                                tgm_pred_n1n1=tgm_pred_n1n1,
                                tgm_pred_n1n2=tgm_pred_n1n2,
                                tgm_pred_n2n1=tgm_pred_n2n1,
                                tgm_pred_n2n2=tgm_pred_n2n2,
                                n1_win_starts=n1_win_starts,
                                n2_win_starts=n2_win_starts,
                                n1_time=n1_time,
                                n2_time=n2_time,
                                proc=proc)