import itertools
import os.path
import numpy as np
import batch_experiments_eos_multisub_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'
NEW_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'

def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(['krns2'], #batch_exp.EXPERIMENTS,
                                   [2], #batch_exp.OVERLAPS,
                                   [False], #batch_exp.IS_PERMS,
                                   ['lr-l2'], #batch_exp.ALGS,
                                   ['zscore'], #batch_exp.ADJS,
                                   [True], #batch_exp.DO_TME_AVGS,
                                   [True],  #batch_exp.DO_TST_AVGS,
                                   [1],  #batch_exp.NUM_INSTANCESS,
                                   [1],  #batch_exp.RANDOM_STATES,
                                   [2],  #batch_exp.WIN_LENS,
                                   ['pooled'],  #batch_exp.SEN_TYPES,
                                   ['verb'])  #batch_exp.WORDS,
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
        win_len = grid[9]
        sen = grid[10]
        word = grid[11]

        # if ni == 1:
        #     continue

        dir_str = batch_exp.JOB_DIR.format(exp=exp)
        top_dir = TOP_DIR.format(exp=exp)
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
        #     continue
        tgm_acc = []
        tgm_pred = []
        cv_membership = []
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
            cv_membership.append(result['cv_membership'][0])
            tgm_acc.append(result['tgm_acc'])
            tgm_pred.append(result['tgm_pred'])

        fold_num = len(batch_exp.FOLDS)
        if exp == 'PassAct3' and word in ['agent', 'patient', 'propid']:
            fold_num /= 2
        if sen in ['active', 'passive']:
            fold_num /= 2

        if len(tgm_acc) == fold_num:
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