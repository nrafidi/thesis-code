import itertools
import os.path
import numpy as np
import batch_experiments_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_{mode}'

NEW_SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_{fold}'


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
                                   batch_exp.DO_TIME_AVGS,
                                   batch_exp.DO_TEST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.REPS_TO_USES,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.SUBJECTS)
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
        reps = grid[8]
        rs = grid[9]
        win_len = grid[10]
        sen = grid[11]
        word = grid[12]
        sub = grid[13]

        dir_str = batch_exp.JOB_DIR.format(exp=exp)
        top_dir = TOP_DIR.format(exp=exp)
        save_dir = SAVE_DIR.format(top_dir=top_dir, sub=sub)
        complete_job = SAVE_FILE.format(dir=save_dir,
                                        sub=sub,
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
                                        rep=reps,
                                        rsP=rs,
                                        mode='acc')
        if os.path.isfile(complete_job + '.npz'):
            continue
        tgm_acc = []
        tgm_pred = []
        for fold in batch_exp.FOLDS:
            fname = NEW_SAVE_FILE.format(dir=save_dir,
                                     sub=sub,
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
                                     rep=reps,
                                     rsP=rs,
                                     fold=fold)

            if not os.path.isfile(fname + '.npz'):
                print('{} missing'.format(fname))
                break

            result = np.load(fname + '.npz')
            if fold == 0:
                l_ints = result['l_ints']
                win_starts = result['win_starts']
                cv_membership = result['cv_membership']
                time = result['time']
                proc = result['proc']
            tgm_acc.append(result['tgm_acc'])
            tgm_pred.append(result['tgm_pred'])

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