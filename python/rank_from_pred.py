import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os.path

OPPOSITES = {0:2,
             1:7,
             2:0,
             3:5,
             4:6,
             5:3,
             6:4,
             7:1}

REGIONS = ['superiorfrontal', 'caudalmiddlefrontal', 'rostralmiddlefrontal', 'parsopercularis', 'parsorbitalis',
              'parstriangularis', 'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole', 'paracentral',
              'precentral', 'insula', 'postcentral', 'inferiorparietal', 'supramarginal', 'superiorparietal',
              'precuneus', 'cuneus', 'lateraloccipital', 'lingual', 'pericalcarine', 'isthmuscingulate',
              'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate', 'entorhinal',
              'parahippocampal', 'temporalpole', 'fusiform', 'superiortemporal', 'inferiortemporal', 'middletemporal',
              'transversetemporal', 'bankssts']

HEMIS = ['lh', 'rh']

def rank_from_pred(tgm_pred, fold_labels):
    rank_acc = np.empty(tgm_pred.shape)
    assert len(fold_labels) == tgm_pred.shape[0]
    for i in range(tgm_pred.shape[0]):
        curr_label = fold_labels[i]
        curr_pred = np.squeeze(tgm_pred[i, ...])
        for j in range(curr_pred.shape[0]):
            for k in range(curr_pred.shape[1]):
                num_items = curr_pred[j, k].shape[0]
                assert num_items == len(curr_label)
                if num_items > 1:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]), axis=1)
                    label_sort = label_sort[:, ::-1]

                    rank = np.empty((num_items,))
                    for l in range(num_items):
                        rank[l] = float(np.where(label_sort[l, :] == curr_label[l])[0][0])

                    rank_acc[i, j, k] = np.mean(1.0 - rank/(float(label_sort.shape[1]) - 1.0))
                else:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]))
                    label_sort = label_sort[::-1]
                    try:
                        rank = float(np.where(label_sort == curr_label)[0][0])
                    except:
                        print(curr_label)
                        print(label_sort)
                        rank = len(label_sort) - 2
                    rank_acc[i, j, k] = 1.0 - rank/(float(len(label_sort)) - 1.0)
                

    return rank_acc


def rank_from_pred_bind(tgm_pred, fold_labels):
    rank_acc = np.empty(tgm_pred.shape)
    assert len(fold_labels) == tgm_pred.shape[0]
    for i in range(tgm_pred.shape[0]):
        curr_label = fold_labels[i]
        curr_pred = np.squeeze(tgm_pred[i, ...])

        for j in range(curr_pred.shape[0]):
            for k in range(curr_pred.shape[1]):

                if curr_pred[j, k].shape[0] > 1:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]), axis=1)
                    # print(label_sort)
                    label_sort = label_sort[:, ::-1]
                    print(label_sort)
                    rank_corr = np.empty((curr_pred[j, k].shape[0],))
                    rank_inc = np.empty((curr_pred[j, k].shape[0],))
                    score = np.empty((curr_pred[j, k].shape[0],))
                    for l in range(curr_pred[j, k].shape[0]):
                        rank_corr[l] = float(np.where(label_sort[l, :] == curr_label[l])[0][0])
                        rank_inc[l] = float(np.where(label_sort[l, :] == OPPOSITES[curr_label[l]])[0][0])
                    # print(rank)
                    # print(label_sort.shape[1])
                        if rank_corr[l] < rank_inc[l]:
                            score[l] = 1.0
                        else:
                            score[l] = 0.0
                    rank_acc[i, j, k] = np.mean(score)
                    # print(rank_acc[i, j, k])
                else:
                    label_sort = np.argsort(np.squeeze(curr_pred[j, k]))
                    label_sort = label_sort[::-1]
                    try:
                        rank_corr = float(np.where(label_sort == curr_label)[0][0])
                        rank_inc = float(np.where(label_sort == OPPOSITES[curr_label])[0][0])
                    except:
                        print(curr_label)
                        print(label_sort)
                        raise
                    if rank_corr < rank_inc:
                        rank_acc[i, j, k] = 1.0
                    else:
                        rank_acc[i, j, k] = 0.0

    return rank_acc



if __name__ == '__main__':

    # fname = '/share/volume0/nrafidi/{exp}_TGM_LOSO/TGM-LOSO_multisub_{sen_type}_{word}_win50_ov5_prF_' \
    #         'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_rsPerm1_{rank_str}acc.npz'
    # for exp in ['krns2', 'PassAct3']:
    #     print(exp)
    #     for sen_type in ['active', 'passive']:
    #         print(sen_type)
    #         for word in ['noun1', 'verb', 'noun2']:
    #             print(word)
    #             fname_load = fname.format(exp=exp, sen_type=sen_type,
    #                                       word=word, rank_str='')
    #             fname_save = fname.format(exp=exp, sen_type=sen_type,
    #                                       word=word, rank_str='rank')
    #             if not os.path.isfile(fname_load):
    #                 continue
    #             if os.path.isfile(fname_save):
    #                 continue
    #             result = np.load(fname_load)
    #             tgm_pred = result['tgm_pred']
    #             l_ints = result['l_ints']
    #             cv_membership = result['cv_membership']
    #             fold_labels = []
    #             for i in range(len(cv_membership)):
    #                 fold_labels.append(np.mean(l_ints[cv_membership[i]]))
    #             tgm_rank = rank_from_pred(tgm_pred, fold_labels)
    #
    #             np.savez_compressed(fname_save,
    #                                 tgm_rank=tgm_rank)

    fname = '/share/volume0/nrafidi/{exp}_TGM_KF_EOS/TGM-LOSO-{k}F_multisub_{sen_type}_{word}_win50_ov5_prF_' \
            'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_' \
            'rsPerm1_rsCV{rsCV}_{rank_str}acc.npz'
    for exp in ['krns2']:
        for num_folds in [2, 4, 8]:
            for rsCV in range(100):
                for sen_type in ['pooled']:
                    print(sen_type)
                    for word in ['verb', 'voice', 'propid']:
                        print(word)
                        if word == 'propid' and sen_type != 'pooled':
                            continue
                        if word == 'propid' and num_folds > 2:
                            continue
                        fname_load = fname.format(exp=exp, sen_type=sen_type,
                                                  rsCV=rsCV, k=num_folds,
                                                  word=word, rank_str='')
                        if not os.path.isfile(fname_load):
                            continue
                        fname_save = fname.format(exp=exp, sen_type=sen_type,
                                                  rsCV=rsCV, k=num_folds,
                                                  word=word, rank_str='rank')

                        result = np.load(fname_load)
                        tgm_pred = result['tgm_pred']
                        l_ints = result['l_ints']
                        cv_membership = result['cv_membership']
                        fold_labels = []
                        for i in range(len(cv_membership)):
                            print(cv_membership[i])
                            fold_labels.append(l_ints[cv_membership[i]])
                        tgm_rank = rank_from_pred(tgm_pred, fold_labels)

                        np.savez_compressed(fname_save,
                                            tgm_rank=tgm_rank)

    # fname = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS_SOURCE/TGM-LOSO-EOS_multisub_{sen_type}_{word}_{reg}_win50_ov5_prF_' \
    #         'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_rsPerm1_{rank_str}acc.npz'
    # for exp in ['PassAct3']:
    #     print(exp)
    #     for hemi in HEMIS:
    #         for reg in REGIONS:
    #             for sen_type in ['pooled']:
    #                 print(sen_type)
    #                 for word in ['verb', 'voice', 'agent', 'patient', 'propid']:
    #                     print(word)
    #                     if word == 'propid' and sen_type != 'pooled':
    #                         continue
    #                     fname_load = fname.format(exp=exp, sen_type=sen_type,
    #                                               reg='{}-{}'.format(reg, hemi),
    #                                               word=word, rank_str='')
    #                     if not os.path.isfile(fname_load):
    #                         continue
    #                     fname_save = fname.format(exp=exp, sen_type=sen_type,
    #                                               reg='{}-{}'.format(reg, hemi),
    #                                               word=word, rank_str='rank')
    #
    #                     result = np.load(fname_load)
    #                     tgm_pred = result['tgm_pred']
    #                     l_ints = result['l_ints']
    #                     cv_membership = result['cv_membership']
    #                     fold_labels = []
    #                     for i in range(len(cv_membership)):
    #                         fold_labels.append(np.mean(l_ints[cv_membership[i]]))
    #                     tgm_rank = rank_from_pred(tgm_pred, fold_labels)
    #
    #                     np.savez_compressed(fname_save,
    #                                         tgm_rank=tgm_rank)
    #
    #                     if word == 'propid':
    #                         fname_save = fname.format(exp=exp, sen_type=sen_type,
    #                                                   reg='{}-{}'.format(reg, hemi),
    #                                                   word='bind', rank_str='rank')
    #                         fname_new_save = fname.format(exp=exp, sen_type=sen_type,
    #                                                       reg='{}-{}'.format(reg, hemi),
    #                                                       word='bind', rank_str='')
    #
    #                         result = np.load(fname_load)
    #                         tgm_pred = result['tgm_pred']
    #                         l_ints = result['l_ints']
    #                         cv_membership = result['cv_membership']
    #                         fold_labels = []
    #                         for i in range(len(cv_membership)):
    #                             fold_labels.append(np.mean(l_ints[cv_membership[i]]))
    #                         tgm_rank = rank_from_pred_bind(tgm_pred, fold_labels)
    #
    #                         np.savez_compressed(fname_new_save,
    #                                             l_ints=result['l_ints'],
    #                                             cv_membership=result['cv_membership'],
    #                                             tgm_acc=result['tgm_acc'],
    #                                             tgm_pred=result['tgm_pred'],
    #                                             win_starts=result['win_starts'],
    #                                             time=result['time'],
    #                                             proc=result['proc'])
    #
    #                         np.savez_compressed(fname_save,
    #                                             tgm_rank=tgm_rank)

    # fname = '/share/volume0/nrafidi/krns2_TGM_alg_comp/TGM-alg-comp_multisub_pooled_verb_win2_ov2_prF_' \
    #         'alglr-l2_adj-zscore_avgTimeT_avgTestF_ni2_rsPerm1_{rank_str}acc.npz'
    #
    # fname_load = fname.format(rank_str='')
    # fname_save = fname.format(rank_str='rank')
    #
    # result = np.load(fname_load)
    # tgm_pred = result['tgm_pred']
    # l_ints = result['l_ints']
    # cv_membership = result['cv_membership']
    # fold_labels = []
    # for i in range(len(cv_membership)):
    #     fold_labels.append(np.mean(l_ints[cv_membership[i]]))
    # tgm_rank = rank_from_pred(tgm_pred, fold_labels)
    #
    # np.savez_compressed(fname_save,
    #                     tgm_rank=tgm_rank)
    # fig, ax = plt.subplots()
    # im = ax.imshow(np.mean(rank_acc, axis=0), interpolation='nearest', vmin=0.5, vmax=1.0)
    # plt.colorbar(im)
    #
    # fig, ax = plt.subplots()
    # ax.plot(np.diag(np.mean(rank_acc, axis=0)))
    # ax.set_ylim([0.0, 1.0])
    # plt.show()
