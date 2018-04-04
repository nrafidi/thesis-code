import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import os
from scipy.stats import spearmanr, kendalltau
import run_Word_RSA
from mpl_toolkits.axes_grid1 import AxesGrid

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_{fig_type}_{sub}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}.pdf'

AGE = {'boy': 'young',
       'girl': 'young',
       'man': 'old',
       'woman': 'old',
}

GEN = {'boy': 'male',
       'girl': 'female',
       'man': 'male',
       'woman': 'female'}


def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    upper_tri_inds = np.triu_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[upper_tri_inds],rdm2[upper_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


def make_model_rdm(labels):
    rdm = np.empty((len(labels), len(labels)))
    for i_lab, lab in enumerate(labels):
        for j_lab in range(i_lab, len(labels)):
            if lab == labels[j_lab]:
                rdm[i_lab, j_lab] = 0
            else:
                rdm[i_lab, j_lab] = 1
            rdm[j_lab, i_lab] = rdm[i_lab, j_lab]
    return rdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--subject')
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='False', choices=['True', 'False'])

    args = parser.parse_args()

    experiment = args.experiment
    subject = args.subject
    win_len = args.win_len
    overlap = args.overlap
    dist = args.dist
    doTimeAvg = run_Word_RSA.str_to_bool(args.doTimeAvg)

    top_dir = run_Word_RSA.TOP_DIR.format(exp=experiment)
    save_dir = run_Word_RSA.SAVE_DIR.format(top_dir=top_dir, sub=subject)


    # voice_scores = []
    # age_scores = []
    # gen_scores = []
    for word in ['noun2', 'det']:
        fname = run_Word_RSA.SAVE_FILE.format(dir=save_dir,
                                            sub=subject,
                                              word=word,
                                             win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_Word_RSA.bool_to_str(doTimeAvg))
        result = np.load(fname + '.npz')
        labels = result['labels']
        voice_labels = result['voice_labels']
        rdm = result['RDM']
        time = result['time'][result['win_starts']]
        if word == 'noun2':
            age_labels = [AGE[lab] for lab in labels]
            gen_labels = [GEN[lab] for lab in labels]
            age_rdm = make_model_rdm(age_labels)
            gen_rdm = make_model_rdm(gen_labels)
        
        voice_rdm = make_model_rdm(voice_labels)

        num_time = rdm.shape[0]
        voice_scores_win = np.empty((num_time,))
        age_scores_win = np.empty((num_time,))
        gen_scores_win = np.empty((num_time,))
        for i_win in range(num_time):
            voice_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), voice_rdm)
            age_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), age_rdm)
            gen_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), gen_rdm)
        # voice_scores.append(voice_scores_win[None, ...])
        # age_scores.append(age_scores_win[None, ...])
        # gen_scores.append(gen_scores_win[None, ...])

        fig, ax = plt.subplots()
        ax.plot(time, voice_scores_win, label='Voice')
        ax.plot(time, age_scores_win, label='Age')
        ax.plot(time, gen_scores_win, label='Gen')
        ax.legend(loc=1)
        ax.set_title('{word} Kendall Tau Scores'.format(word=word), fontsize=18)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Kendall Tau Correlation')
        fig.savefig(SAVE_FIG.format(fig_type='score-overlay',
                                    sub=subject,
                                    word=word,
                                    win_len=win_len,
                                    ov=overlap,
                                    dist=dist,
                                    avgTm=run_Word_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

        print(word)
        best_voice_win = np.argmax(voice_scores_win)
        print('Best Voice Correlation occurs at {}'.format(time[best_voice_win]))
        best_voice_score = voice_scores_win[best_voice_win]
        best_age_win = np.argmax(age_scores_win)
        print('Best Age Correlation occurs at {}'.format(time[best_age_win]))
        best_age_score = age_scores_win[best_age_win]
        best_gen_win = np.argmax(gen_scores_win)
        print('Best Gen Correlation occurs at {}'.format(time[best_gen_win]))
        best_gen_score = gen_scores_win[best_gen_win]

        voice_fig = plt.figure(figsize=(14, 7))
        voice_grid = AxesGrid(voice_fig, 111, nrows_ncols=(1, 2),
                                  axes_pad=0.4, cbar_mode='single', cbar_location='right',
                                  cbar_pad=0.4)
        voice_grid[0].imshow(voice_rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
        voice_grid[0].set_title('Model', fontsize=14)
        voice_grid[0].text(-0.15, 1.0, 'A', transform=voice_grid[0].transAxes,
                                            size=20, weight='bold')
        im = voice_grid[1].imshow(np.squeeze(rdm[best_voice_win, ...]), interpolation='nearest', vmin=0.0, vmax=1.0)
        voice_grid[1].set_title('MEG', fontsize=14)
        voice_grid[1].text(-0.15, 1.0, 'B', transform=voice_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = voice_grid.cbar_axes[0].colorbar(im)
        voice_fig.suptitle('Voice {word} RDM Comparison\nScore: {score}'.format(word=word,
                                                                          score=best_voice_score),
                     fontsize=18)

        voice_fig.savefig(SAVE_FIG.format(fig_type='voice-rdm',
                                            sub=subject,
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_Word_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

        age_fig = plt.figure(figsize=(14, 7))
        age_grid = AxesGrid(age_fig, 111, nrows_ncols=(1, 2),
                              axes_pad=0.4, cbar_mode='single', cbar_location='right',
                              cbar_pad=0.4)
        age_grid[0].imshow(age_rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
        age_grid[0].set_title('Model', fontsize=14)
        age_grid[0].text(-0.15, 1.0, 'A', transform=age_grid[0].transAxes,
                           size=20, weight='bold')
        im = age_grid[1].imshow(np.squeeze(rdm[best_age_win, ...]), interpolation='nearest', vmin=0.0, vmax=1.0)
        age_grid[1].set_title('MEG', fontsize=14)
        age_grid[1].text(-0.15, 1.0, 'B', transform=age_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = age_grid.cbar_axes[0].colorbar(im)
        age_fig.suptitle('Age {word} RDM Comparison\nScore: {score}'.format(word=word,
                                                                          score=best_age_score),
                     fontsize=18)

        age_fig.savefig(SAVE_FIG.format(fig_type='age-rdm',
                                          sub=subject,
                                          word=word,
                                          win_len=win_len,
                                          ov=overlap,
                                          dist=dist,
                                          avgTm=run_Word_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

        gen_fig = plt.figure(figsize=(14, 7))
        gen_grid = AxesGrid(gen_fig, 111, nrows_ncols=(1, 2),
                              axes_pad=0.4, cbar_mode='single', cbar_location='right',
                              cbar_pad=0.4)
        gen_grid[0].imshow(gen_rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
        gen_grid[0].set_title('Model', fontsize=14)
        gen_grid[0].text(-0.15, 1.0, 'A', transform=gen_grid[0].transAxes,
                           size=20, weight='bold')
        im = gen_grid[1].imshow(np.squeeze(rdm[best_gen_win, ...]), interpolation='nearest', vmin=0.0, vmax=1.0)
        gen_grid[1].set_title('MEG', fontsize=14)
        gen_grid[1].text(-0.15, 1.0, 'B', transform=gen_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = gen_grid.cbar_axes[0].colorbar(im)
        gen_fig.suptitle('Gender {word} RDM Comparison\nScore: {score}'.format(word=word,
                                                                          score=best_gen_score),
                     fontsize=18)

        gen_fig.savefig(SAVE_FIG.format(fig_type='gen-rdm',
                                          sub=subject,
                                          word=word,
                                          win_len=win_len,
                                          ov=overlap,
                                          dist=dist,
                                          avgTm=run_Word_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')
    plt.show()