import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def rank_from_pred(tgm_pred, fold_labels):
    rank_acc = np.empty(tgm_pred.shape)
    for i in range(tgm_pred.shape[0]):
        curr_label = fold_labels[i]
        curr_pred = np.squeeze(tgm_pred[i, ...])
        for j in range(curr_pred.shape[0]):
            for k in range(curr_pred.shape[0]):
                label_sort = np.argsort(np.squeeze(curr_pred[j, k]))
                # print(curr_pred[j, k])
                # print(label_sort)
                label_sort = label_sort[::-1]
                rank = float(np.where(label_sort == curr_label)[0][0])
                rank_acc[i, j, k] = 1.0 - rank/(float(len(label_sort)) - 1.0)

    return rank_acc


if __name__ == '__main__':
    result = np.load('/share/volume0/nrafidi/PassAct3_TGM_LOSO/TGM-LOSO_multisub_passive_noun2_win50_ov5_prF_' \
            'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_rsPerm1_acc.npz')
    tgm_pred = result['tgm_pred']
    l_ints = result['l_ints']
    # print(l_ints)
    cv_membership = result['cv_membership']
    # print(len(cv_membership))
    fold_labels = []
    for i in range(len(cv_membership)):
        fold_labels.append(np.mean(l_ints[cv_membership[i]]))
    # print(cv_membership)
    rank_acc = rank_from_pred(tgm_pred, fold_labels)
    fig, ax = plt.subplots()
    im = ax.imshow(np.mean(rank_acc, axis=0), interpolation='nearest', vmin=0.5, vmax=1.0)
    plt.colorbar(im)

    fig, ax = plt.subplots()
    ax.plot(np.diag(np.mean(rank_acc, axis=0)))
    ax.set_ylim([0.0, 1.0])
    plt.show()
