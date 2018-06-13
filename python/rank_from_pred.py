import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os.path

def rank_from_pred(tgm_pred, fold_labels):
    rank_acc = np.empty(tgm_pred.shape)
    for i in range(tgm_pred.shape[0]):
        curr_label = fold_labels[i]
#        print(curr_label)
        curr_pred = np.squeeze(tgm_pred[i, ...])
        for j in range(curr_pred.shape[0]):
            for k in range(curr_pred.shape[0]):
                if curr_pred[j, k].shape[0] > 1:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]), axis=1)
                    print(label_sort)
                    label_sort = label_sort[:, ::-1]
                    print(label_sort)
                    rank = np.empty((curr_pred[j, k].shape[0],))
                    for l in range(curr_pred[j, k].shape[0]):
                        rank[l] = float(np.where(label_sort[l, :] == curr_label)[0][0])
                    print(rank)
                    rank_acc[i, j, k] = np.mean(1.0 - rank/(float(label_sort.shape[1]) - 1.0))

                else:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]))
                    label_sort = label_sort[::-1]
                    rank = float(np.where(label_sort == curr_label)[0][0])
                    rank_acc[i, j, k] = 1.0 - rank/(float(len(label_sort)) - 1.0)
                

    return rank_acc


if __name__ == '__main__':

    fname = '/share/volume0/nrafidi/{exp}_TGM_LOSO/TGM-LOSO_multisub_{sen_type}_{word}_win50_ov5_prF_' \
            'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_rsPerm1_{rank_str}acc.npz'
    for exp in ['krns2', 'PassAct3']:
        print(exp)
        for sen_type in ['active', 'passive']:
            print(sen_type)
            for word in ['noun1', 'verb', 'noun2']:
                print(word)
                fname_load = fname.format(exp=exp, sen_type=sen_type,
                                          word=word, rank_str='')
                fname_save = fname.format(exp=exp, sen_type=sen_type,
                                          word=word, rank_str='rank')
                if not os.path.isfile(fname_load):
                    continue
                if os.path.isfile(fname_save):
                    continue
                result = np.load(fname_load)
                tgm_pred = result['tgm_pred']
                l_ints = result['l_ints']
                cv_membership = result['cv_membership']
                fold_labels = []
                for i in range(len(cv_membership)):
                    fold_labels.append(np.mean(l_ints[cv_membership[i]]))
                tgm_rank = rank_from_pred(tgm_pred, fold_labels)

                np.savez_compressed(fname_save,
                                    tgm_rank=tgm_rank)

    fname = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/TGM-LOSO-EOS_multisub_{sen_type}_{word}_win50_ov5_prF_' \
            'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_rsPerm1_{rank_str}acc.npz'
    for exp in ['krns2', 'PassAct3']:
        print(exp)
        for sen_type in ['active', 'passive', 'pooled']:
            print(sen_type)
            for word in ['noun1', 'verb', 'agent', 'patient', 'voice', 'propid', 'senlen']:
                print(word)
                if word == 'propid' and sen_type != 'pooled':
                    continue
                fname_load = fname.format(exp=exp, sen_type=sen_type,
                                          word=word, rank_str='')
                fname_save = fname.format(exp=exp, sen_type=sen_type,
                                          word=word, rank_str='rank')
                if not os.path.isfile(fname_load):
                    continue
                if os.path.isfile(fname_save):
                    continue
                result = np.load(fname_load)
                tgm_pred = result['tgm_pred']
                l_ints = result['l_ints']
                cv_membership = result['cv_membership']
                fold_labels = []
                for i in range(len(cv_membership)):
                    fold_labels.append(np.mean(l_ints[cv_membership[i]]))
                tgm_rank = rank_from_pred(tgm_pred, fold_labels)

                np.savez_compressed(fname_save,
                                    tgm_rank=tgm_rank)

    fname = '/share/volume0/nrafidi/krns2_TGM_alg_comp/TGM-alg-comp_multisub_pooled_verb_win2_ov2_prF_' \
            'alglr-l2_adj-zscore_avgTimeT_avgTestF_ni2_rsPerm1_{rank_str}acc.npz'

    fname_load = fname.format(rank_str='')
    fname_save = fname.format(rank_str='rank')

    result = np.load(fname_load)
    tgm_pred = result['tgm_pred']
    l_ints = result['l_ints']
    cv_membership = result['cv_membership']
    fold_labels = []
    for i in range(len(cv_membership)):
        fold_labels.append(np.mean(l_ints[cv_membership[i]]))
    tgm_rank = rank_from_pred(tgm_pred, fold_labels)

    np.savez_compressed(fname_save,
                        tgm_rank=tgm_rank)
    # fig, ax = plt.subplots()
    # im = ax.imshow(np.mean(rank_acc, axis=0), interpolation='nearest', vmin=0.5, vmax=1.0)
    # plt.colorbar(im)
    #
    # fig, ax = plt.subplots()
    # ax.plot(np.diag(np.mean(rank_acc, axis=0)))
    # ax.set_ylim([0.0, 1.0])
    # plt.show()
