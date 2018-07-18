import itertools
import os.path
import numpy as np
import batch_experiments_eos_multisub_fold_source as batch_exp

TOP_DIR = '/share/volume0/nrafidi/PassAct3_TGM_LOSO_EOS_SOURCE/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_{reg}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'
NEW_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_{reg}_win{win_len}_ov{ov}_pr{perm}_' \
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
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.DO_TST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.HEMIS,
                                   batch_exp.REGIONS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.FOLDS)
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
        hemi = grid[9]
        reg = grid[10]
        sen = grid[11]
        word = grid[12]
        fold = grid[13]

        if word in ['propid', 'voice', 'senlen', 'noun1'] and sen != 'pooled':
            continue

        fold_num = len(batch_exp.FOLDS)
        if word in ['agent', 'patient', 'propid']:
            fold_num /= 2
        if sen in ['active', 'passive']:
            fold_num /= 2
        print('meow')
        top_dir = TOP_DIR
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
                                 reg='{}-{}'.format(reg, hemi),
                                 mode='acc')

        if os.path.isfile(complete_job + '.npz'):
            job_id += fold_num
            continue
        tgm_acc = []
        tgm_pred = []
        cv_membership = []
        for fold in range(fold_num):
            print('woof')
            if fold > 15 and sen != 'pooled':
                break
            if word in ['agent', 'patient', 'propid']:
                if sen != 'pooled' and fold > 7:
                    break
                elif sen == 'pooled' and fold > 15:
                    break
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
                                 reg='{}-{}'.format(reg, hemi),
                                 fold=fold)

            if not os.path.isfile(fname + '.npz'):
                print('{} missing'.format(fname))
                print(job_id)
                job_id += fold_num - fold
                break

            try:
                result = np.load(fname + '.npz')
            except:
                print('{} needs to be rerun'.format(fname))
                print(job_id)
                job_id += fold_num - fold - 1
                break
            if fold == 0:
                l_ints = result['l_ints']
                win_starts = result['win_starts']
                time = result['time']
                proc = result['proc']
            cv_membership.append(result['cv_membership'][0])
            tgm_acc.append(result['tgm_acc'])
            tgm_pred.append(result['tgm_pred'])
            job_id += 1

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