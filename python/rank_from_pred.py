import numpy as np

def rank_from_pred(tgm_pred, fold_labels):
    print(tgm_pred.shape)
    for i in range(tgm_pred.shape[0]):
        print(type(tgm_pred[i]))
        #for j in range(tgm_pred.shape[1]):
           # print(tgm_pred[i, j, j].shape)


if __name__ == '__main__':
    result = np.load('/share/volume0/nrafidi/krns2_TGM_LOSO_EOS/TGM-LOSO-EOS_multisub_active_agent_win50_ov5_prF_' \
            'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_rsPerm1_acc.npz')
    tgm_pred = result['tgm_pred']
    l_ints = result['l_ints']
    print(l_ints)
    cv_membership = result['cv_membership']
    print(len(cv_membership))
    fold_labels = []
    for i in range(len(cv_membership)/2):
        fold_labels.append(l_ints)
    print(cv_membership)
    rank_from_pred(tgm_pred, l_ints)
