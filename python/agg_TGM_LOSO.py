import argparse
import load_data_ordered as load_data
import numpy as np
import os.path
import run_TGM_LOSO


AGG_FILE = '{dir}TGM-LOSO_{sen_type}_{word}_win{win_len}_ov{ov}_prF_' \
           'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
           'nr10_rsPerm1.npz'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_TGM_LOSO.VALID_SEN_TYPE)
    args = parser.parse_args()

    exp = args.experiment
    sen_type = args.sen_type

    top_dir = run_TGM_LOSO.TOP_DIR.format(exp=exp)
    total_agg = '{top_dir}agg_file.npz'.format(top_dir=top_dir)

    if os.path.isfile(total_agg):
        tgm_by_word = np.load(total_agg)
    else:
        tgm_by_word = []
        for word in ['noun1', 'verb', 'noun2']:
            tgm_by_win = []
            for win_len in [12, 25, 50]:
                tgm_by_ov = []
                for overlap in [12, 25, 50]:
                    tgm_by_alg = []
                    for alg in ['lr-l2', 'lr-l1']:
                        tgm_by_adj = []
                        for adj in [None, 'zscore']:
                            tgm_by_inst = []
                            for inst in [1, 2, 5, 10]:
                                tgm_by_avgTm = []
                                for avgTm in ['T', 'F']:
                                    tgm_by_avgTst = []
                                    for avgTst in ['T', 'F']:
                                        agg = AGG_FILE.format(dir=top_dir,
                                                              sen_type=sen_type,
                                                              word=word,win_len=win_len,
                                                              overlap=overlap,
                                                              alg=alg,
                                                              adj=adj,
                                                              inst=inst,
                                                              avgTm=avgTm,
                                                              avgTst=avgTst)
                                        if os.path.isfile(agg):
                                            result = np.load(agg)
                                            tgm_by_avgTst.append(result['tgm'][None, ...])
                                        else:
                                            tgm_by_sub = []
                                            for sub in load_data.VALID_SUBS[exp]:
                                                save_dir = run_TGM_LOSO.SAVE_DIR.format(sub=sub)
                                                result = np.load(run_TGM_LOSO.SAVE_FILE.format(dir=save_dir,
                                                                                               sub=sub,
                                                                                               sen_type=sen_type,
                                                                                               word=word,
                                                                                               win_len=win_len,
                                                                                               ov=overlap,
                                                                                               parm='F',
                                                                                               alg=alg,
                                                                                               adj=adj,
                                                                                               avgTm=avgTm,
                                                                                               avgTst=avgTst,
                                                                                               inst=inst,
                                                                                               rep=10,
                                                                                               rsP=1))
                                                tgm = np.mean(result['tgm'], axis=0)
                                                np.savez(agg, tgm=tgm)
                                                tgm_by_sub.append(tgm[None, ...])
                                            tgm_by_sub = np.mean(np.concatenate(tgm_by_sub, axis=0), axis=0)
                                            tgm_by_avgTst.append(tgm_by_sub[None, ...])
                                    tgm_by_avgTst = np.concatenate(tgm_by_avgTst)
                                    tgm_by_avgTm.append(tgm_by_avgTst[None, ...])
                                tgm_by_avgTm = np.concatenate(tgm_by_avgTm)
                                tgm_by_inst.append(tgm_by_avgTm[None, ...])
                            tgm_by_inst = np.concatenate(tgm_by_inst)
                            tgm_by_adj.append(tgm_by_adj[None, ...])
                        tgm_by_adj = np.concatenate(tgm_by_adj)
                        tgm_by_alg.append(tgm_by_alg[None, ...])
                    tgm_by_alg = np.concatenate(tgm_by_alg)
                    tgm_by_ov.append(tgm_by_ov[None, ...])
                tgm_by_ov = np.concatenate(tgm_by_ov)
                tgm_by_win.append(tgm_by_win[None, ...])
            tgm_by_win = np.concatenate(tgm_by_win)
            tgm_by_word.append(tgm_by_word[None, ...])
        tgm_by_word = np.concatenate(tgm_by_word)

        np.savez(total_agg, tgm=tgm_by_word)
