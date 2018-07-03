import itertools
import os.path
import numpy as np
import batch_experiments_eos_multisub_subcv_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub-less{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'

TOTAL_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub-subcv_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'
NEW_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub-less{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'

NUM_SUBS = {'krns2': 8,
            'PassAct3': 20}


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(['krns2'], #batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.DO_TST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
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
        rs = grid[8]
        win_len = grid[9]
        sen = grid[10]
        word = grid[11]

        if exp == 'krns2':
            if word == 'senlen':
                continue
        if sen != 'pooled':
            if word in ['noun1', 'voice', 'senlen', 'propid']:
                continue

        if sen == 'active' and word == 'verb':
            print('wtf')
        dir_str = batch_exp.JOB_DIR.format(exp=exp)
        top_dir = TOP_DIR.format(exp=exp)
        total_job = TOTAL_SAVE_FILE.format(dir=top_dir,
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
        if os.path.isfile(total_job + '.npz'):
            if sen == 'active' and word == 'verb':
                print(total_job)
            continue
        tgm_acc = []
        tgm_pred = []
        cv_membership = []
        for i_sub, sub in enumerate(batch_exp.SUBJECTS):
            if sub not in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                continue
            complete_job = SAVE_FILE.format(dir=top_dir,
                                            sen_type=sen,
                                            word=word,
                                            win_len=win_len,
                                            sub=sub,
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
                total_result = np.load(complete_job + '.npz')
                l_ints_sub = total_result['l_ints']
                win_starts_sub = total_result['win_starts']

                time_sub = total_result['time']
                proc_sub = total_result['proc']
                cv_membership.append(total_result['cv_membership'][0])
                tgm_acc.append(total_result['tgm_acc'])
                tgm_pred.append(total_result['tgm_pred'])
                continue
            tgm_acc_sub = []
            tgm_pred_sub = []
            cv_membership_sub = []
            for fold in batch_exp.FOLDS:
                fname = NEW_SAVE_FILE.format(dir=top_dir,
                                     sen_type=sen,
                                     word=word,
                                     win_len=win_len,
                                             sub=sub,
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
                    l_ints_sub = result['l_ints']
                    win_starts_sub = result['win_starts']

                    time_sub = result['time']
                    proc_sub = result['proc']
                cv_membership_sub.append(result['cv_membership'][0])
                tgm_acc_sub.append(result['tgm_acc'])
                tgm_pred_sub.append(result['tgm_pred'])

            fold_num = len(batch_exp.FOLDS)
            if exp == 'PassAct3' and word in ['agent', 'patient', 'propid']:
                fold_num /= 2
            if sen in ['active', 'passive']:
                fold_num /= 2

            if len(tgm_acc_sub) == fold_num:
                tgm_acc_sub = np.concatenate(tgm_acc_sub, axis=0)
                tgm_pred_sub= np.concatenate(tgm_pred_sub, axis=0)

                print(tgm_acc_sub.shape)

                np.savez_compressed(complete_job + '.npz',
                                    l_ints=l_ints_sub,
                                    cv_membership=cv_membership_sub,
                                    tgm_acc=tgm_acc_sub,
                                    tgm_pred=tgm_pred_sub,
                                    win_starts=win_starts_sub,
                                    time=time_sub,
                                    proc=proc_sub)
                tgm_acc.append(tgm_acc_sub[None, ...])
                tgm_pred.append(tgm_pred_sub[None, ...])
                cv_membership.append(cv_membership_sub)

                # if i_sub == 0:
                #     l_ints = l_ints_sub
                #     win_starts = win_starts_sub
                #     time = time_sub
                #     proc = proc_sub
        if len(tgm_acc) == NUM_SUBS[exp]:
            tgm_acc = np.concatenate(tgm_acc, axis=0)
            tgm_pred = np.concatenate(tgm_pred, axis=0)

            print(tgm_acc.shape)

            np.savez_compressed(total_job + '.npz',
                                l_ints=l_ints_sub,
                                cv_membership=cv_membership,
                                tgm_acc=tgm_acc,
                                tgm_pred=tgm_pred,
                                win_starts=win_starts_sub,
                                time=time_sub,
                                proc=proc_sub)