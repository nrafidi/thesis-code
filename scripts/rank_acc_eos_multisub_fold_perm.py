import itertools
import os.path
import numpy as np
import batch_experiments_eos_multisub_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'

SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'


def rank_from_pred(tgm_pred, fold_labels):
    rank_acc = np.empty(tgm_pred.shape)
    print(len(fold_labels))
    print(tgm_pred.shape)
    assert len(fold_labels) == tgm_pred.shape[0]
    for i in range(tgm_pred.shape[0]):
        curr_label = fold_labels[i]
        #        print(curr_label)
        curr_pred = np.squeeze(tgm_pred[i, ...])
        for j in range(curr_pred.shape[0]):
            for k in range(curr_pred.shape[1]):
                if curr_pred[j, k].shape[0] > 1:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]), axis=1)
                    # print(label_sort)
                    label_sort = label_sort[:, ::-1]
                    # print(label_sort)
                    rank = np.empty((curr_pred[j, k].shape[0],))
                    for l in range(curr_pred[j, k].shape[0]):
                        rank[l] = float(np.where(label_sort[l, :] == curr_label)[0][0])
                    # print(rank)
                    # print(label_sort.shape[1])
                    rank_acc[i, j, k] = np.mean(1.0 - rank / (float(label_sort.shape[1]) - 1.0))
                    # print(rank_acc[i, j, k])
                else:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]))
                    label_sort = label_sort[::-1]
                    rank = float(np.where(label_sort == curr_label)[0][0])
                    rank_acc[i, j, k] = 1.0 - rank / (float(len(label_sort)) - 1.0)

    return rank_acc


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   [True],
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.DO_TST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS)
    perm_list = range(200)
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

        complete_job_perm_rank = SAVE_FILE.format(dir=top_dir,
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
                                             mode='rankacc')

        if not os.path.isfile(complete_job_perm_rank + '.npz'):
            result = np.load(complete_job_perm)
            tgm_pred = result['tgm_pred']
            num_perm = tgm_pred.shape[0]
            l_ints = result['l_ints']
            cv_membership_all = result['cv_membership']
            multi_fold_acc = []
            for i_perm in range(num_perm):
                cv_membership = cv_membership_all[i_perm]
                fold_labels = []
                for i in range(len(cv_membership)):
                    fold_labels.append(np.mean(l_ints[cv_membership[i]]))

                tgm_rank = rank_from_pred(tgm_pred[i_perm, ...], fold_labels)
                multi_fold_acc.append(tgm_rank[None, ...])
            multi_fold_acc = np.concatenate(multi_fold_acc, axis=0)
            print(multi_fold_acc.shape)
            np.savez_compressed(complete_job_perm_rank, tgm_rank=multi_fold_acc)