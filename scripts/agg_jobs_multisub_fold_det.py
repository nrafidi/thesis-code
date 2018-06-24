import itertools
import os.path
import numpy as np
import batch_experiments_det_multisub_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/krns2_TGM_LOSO_det/'
SAVE_FILE = '{dir}TGM-LOSO-det_multisub_{sen_type}_{analysis}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'
NEW_SAVE_FILE = '{dir}TGM-LOSO-det_multisub_{sen_type}_{analysis}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TIME_AVGS,
                                   batch_exp.DO_TEST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.ANALYSES)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        overlap = grid[0]
        isPerm = grid[1]
        alg = grid[2]
        adj = grid[3]
        tm_avg = grid[4]
        tst_avg = grid[5]
        ni = grid[6]
        rs = grid[7]
        win_len = grid[8]
        sen = grid[9]
        analysis = grid[10]

        complete_job = SAVE_FILE.format(dir=TOP_DIR,
                                        sen_type=sen,
                                        analysis=analysis,
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
        tgm_acc = []
        tgm_pred = []
        cv_membership = []
        for fold in batch_exp.FOLDS:
            fname = NEW_SAVE_FILE.format(dir=TOP_DIR,
                             sen_type=sen,
                             analysis=analysis,
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
            cv_membership.append(result['cv_membership'][0])
            tgm_acc.append(result['tgm_acc'])
            tgm_pred.append(result['tgm_pred'])

        num_folds = len(batch_exp.FOLDS)
        if 'dog' in analysis:
            num_folds = 26
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
                                time=time)