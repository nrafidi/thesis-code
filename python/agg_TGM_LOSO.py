import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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

    words = ['noun1', 'verb', 'noun2']
    time_lens = [12]#, 25, 50]
    algs = ['lr-l1']#, 'lr-l1']
    adjs=[None]#, 'zscore']
    insts = [1]#, 2, 5, 10]
    bools = ['T']#, 'F']

    if not os.path.isfile(total_agg):
        tgm_by_win = []
        for win_len in time_lens:
            tgm_by_ov = []
            for overlap in time_lens:
                tgm_by_alg = []
                for alg in algs:
                    tgm_by_adj = []
                    for adj in adjs:
                        tgm_by_inst = []
                        for inst in insts:
                            tgm_by_avgTm = []
                            for avgTm in bools:
                                tgm_by_avgTst = []
                                for avgTst in bools:
                                    tgm_by_word = []
                                    for word in words:
                                        agg = AGG_FILE.format(dir=top_dir,
                                                              sen_type=sen_type,
                                                              word=word,
                                                              win_len=win_len,
                                                              ov=overlap,
                                                              alg=alg,
                                                              adj=adj,
                                                              inst=inst,
                                                              avgTm=avgTm,
                                                              avgTst=avgTst)
                                        if os.path.isfile(agg):
                                            result = np.load(agg)
                                            tgm_by_word.append(result['tgm'][None, ...])
                                        else:
                                            tgm_by_sub = []
                                            for sub in load_data.VALID_SUBS[exp]:
                                                save_dir = run_TGM_LOSO.SAVE_DIR.format(top_dir=top_dir, sub=sub)
                                                result = np.load(run_TGM_LOSO.SAVE_FILE.format(dir=save_dir,
                                                                                               sub=sub,
                                                                                               sen_type=sen_type,
                                                                                               word=word,
                                                                                               win_len=win_len,
                                                                                               ov=overlap,
                                                                                               perm='F',
                                                                                               alg=alg,
                                                                                               adj=adj,
                                                                                               avgTm=avgTm,
                                                                                               avgTst=avgTst,
                                                                                               inst=inst,
                                                                                               rep=10,
                                                                                               rsP=1) + '.npz')
                                                print(result.keys())
                                                tgm = np.mean(result['tgm_acc'], axis=0)
                                                tgm_by_sub.append(tgm[None, ...])
                                            tgm_by_sub = np.mean(np.concatenate(tgm_by_sub, axis=0), axis=0)
                                            np.savez(agg, tgm=tgm_by_sub)
                                            tgm_by_word.append(tgm_by_sub[None, ...])
                                    tgm_by_word = np.concatenate(tgm_by_word)
                                    tgm_by_avgTst.append(tgm_by_word[None, ...])
                                tgm_by_avgTst = np.concatenate(tgm_by_avgTst)
                                tgm_by_avgTm.append(tgm_by_avgTst[None, ...])
                            tgm_by_avgTm = np.concatenate(tgm_by_avgTm)
                            tgm_by_inst.append(tgm_by_avgTm[None, ...])
                        tgm_by_inst = np.concatenate(tgm_by_inst)
                        tgm_by_adj.append(tgm_by_inst[None, ...])
                    tgm_by_adj = np.concatenate(tgm_by_adj)
                    tgm_by_alg.append(tgm_by_adj[None, ...])
                tgm_by_alg = np.concatenate(tgm_by_alg)
                tgm_by_ov.append(tgm_by_alg[None, ...])
            tgm_by_ov = np.concatenate(tgm_by_ov)
            tgm_by_win.append(tgm_by_ov[None, ...])
        tgm_by_win = np.concatenate(tgm_by_win)
        print(tgm_by_win.shape)
        np.savez(total_agg, tgm=tgm_by_win)
    else:
        result = np.load(total_agg)
        tgm_by_win = result['tgm']

    tgm = np.squeeze(tgm_by_win)
    fig, ax = plt.subplots()
    ax.plot(np.diag(tgm[0, :]), color='r')
    ax.plot(np.diag(tgm[1, :]), color='b')
    ax.plot(np.diag(tgm[2, :]), color='g')

    plt.show()
