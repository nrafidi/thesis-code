import numpy as np

def rank_from_pred(tgm_pred):
    print(tgm_pred.shape)
    for i in range(tgm_pred.shape[0]):
        print(tgm_pred[i, ...])
        for j in range(tgm_pred.shape[1]):
            print(tgm_pred[i, j, ...])


if __name__ == '__main__':
    result = np.load('/share/volume0/nrafidi/krns2_TGM_LOSO_EOS/TGM-LOSO-EOS_multisub_active_agent_win50_ov5_prF_' \
            'alglr-l2_adj-zscore_avgTimeT_avgTestT_ni2_rsPerm1_acc.npz')
    tgm_pred = result['tgm_pred']
    rank_from_pred(tgm_pred)