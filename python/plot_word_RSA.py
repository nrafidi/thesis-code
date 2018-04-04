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
import string

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_{fig_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}.pdf'

AGE = {'boy': 'young',
       'girl': 'young',
       'man': 'old',
       'woman': 'old',
}

GEN = {'boy': 'male',
       'girl': 'female',
       'man': 'male',
       'woman': 'female'}

PLOT_TITLE = {'det': 'Determiner',
              'noun2': 'Second Noun'}


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


def load_all_rdms(experiment, word, win_len, overlap, dist, avgTm):
    top_dir = run_Word_RSA.TOP_DIR.format(exp=experiment)
    subject_rdms = []
    for i_subject, subject in enumerate(run_Word_RSA.VALID_SUBS[experiment]):
        save_dir = run_Word_RSA.SAVE_DIR.format(top_dir=top_dir, sub=subject)
        fname = run_Word_RSA.SAVE_FILE.format(dir=save_dir,
                                              sub=subject,
                                              word=word,
                                              win_len=win_len,
                                              ov=overlap,
                                              dist=dist,
                                              avgTm=avgTm)
        result = np.load(fname + '.npz')
        new_labels = result['labels']
        new_voice_labels = result['voice_labels']
        if i_subject == 0:
            labels = new_labels
            voice_labels = new_voice_labels

        assert np.all(np.array(new_labels) == np.array(labels))
        assert np.all(np.array(new_voice_labels) == np.array(voice_labels))

        subject_rdms.append(result['RDM'][None, ...])
        time = result['time'][result['win_starts']]
    if word == 'det':
        time += 0.5
    if word == 'noun2':
        age_labels = [AGE[lab] for lab in labels]
        gen_labels = [GEN[lab] for lab in labels]
        age_rdm = make_model_rdm(age_labels)
        gen_rdm = make_model_rdm(gen_labels)
    else:
        age_rdm = []
        gen_rdm = []

    voice_rdm = make_model_rdm(voice_labels)
    return np.concatenate(subject_rdms, axis=0), voice_rdm, age_rdm, gen_rdm, time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='False', choices=['True', 'False'])

    args = parser.parse_args()

    experiment = args.experiment
    win_len = args.win_len
    overlap = args.overlap
    dist = args.dist
    doTimeAvg = run_Word_RSA.str_to_bool(args.doTimeAvg)




    # voice_scores = []
    # age_scores = []
    # gen_scores = []
    score_fig = plt.figure(figsize=(12, 9))
    score_grid = AxesGrid(score_fig, 111, nrows_ncols=(1, 2),
                        axes_pad=0.4, cbar_pad=0.4)
    for i_word, word in enumerate(['noun2', 'det']):
        if word == 'noun2':
            subject_rdms, voice_rdm, age_rdm, gen_rdm, time = load_all_rdms(experiment,
                                                                              word,
                                                                              win_len,
                                                                              overlap,
                                                                              dist,
                                                                              run_Word_RSA.bool_to_str(doTimeAvg))
        else:
            subject_rdms, _, _, _, time = load_all_rdms(experiment,
                                                        word,
                                                        win_len,
                                                        overlap,
                                                        dist,
                                                        run_Word_RSA.bool_to_str(doTimeAvg))
        rdm = np.squeeze(np.mean(subject_rdms, axis=0))
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
        ax = score_grid[1 - i_word]
        ax.plot(time, voice_scores_win, label='Voice')
        ax.plot(time, age_scores_win, label='Age')
        ax.plot(time, gen_scores_win, label='Gen')
        ax.legend(loc=1)
        ax.set_title('{word}'.format(word=PLOT_TITLE[word]), fontsize=14)
        ax.set_xlabel('Time (s)')
        if i_word == 1:
            ax.set_ylabel('Kendall Tau Correlation')
        ax.set_ylim([0.0, 0.7])
        ax.text(-0.11, 1.0, string.ascii_uppercase[1 - i_word], transform=ax.transAxes,
                           size=20, weight='bold')

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
        voice_grid[0].text(-0.1, 1.0, 'A', transform=voice_grid[0].transAxes,
                                            size=20, weight='bold')
        im = voice_grid[1].imshow(np.squeeze(rdm[best_voice_win, ...]), interpolation='nearest', vmin=0.0, vmax=1.0)
        # print(np.squeeze(rdm[best_voice_win, ...]))
        voice_grid[1].set_title('MEG', fontsize=14)
        voice_grid[1].text(-0.09, 1.0, 'B', transform=voice_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = voice_grid.cbar_axes[0].colorbar(im)
        voice_fig.suptitle('Voice {word} RDM Comparison\nScore: {score}'.format(word=word,
                                                                          score=best_voice_score),
                     fontsize=18)

        voice_fig.savefig(SAVE_FIG.format(fig_type='voice-rdm',
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
        age_grid[0].text(-0.1, 1.0, 'A', transform=age_grid[0].transAxes,
                           size=20, weight='bold')
        im = age_grid[1].imshow(np.squeeze(rdm[best_age_win, ...]), interpolation='nearest', vmin=0.0, vmax=1.0)
        # print(np.squeeze(rdm[best_age_win, ...]))
        age_grid[1].set_title('MEG', fontsize=14)
        age_grid[1].text(-0.09, 1.0, 'B', transform=age_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = age_grid.cbar_axes[0].colorbar(im)
        age_fig.suptitle('Age {word} RDM Comparison\nScore: {score}'.format(word=word,
                                                                          score=best_age_score),
                     fontsize=18)

        age_fig.savefig(SAVE_FIG.format(fig_type='age-rdm',
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
        gen_grid[0].text(-0.1, 1.0, 'A', transform=gen_grid[0].transAxes,
                           size=20, weight='bold')
        im = gen_grid[1].imshow(np.squeeze(rdm[best_gen_win, ...]), interpolation='nearest', vmin=0.0, vmax=1.0)
        # print(np.squeeze(rdm[best_gen_win, ...]))
        gen_grid[1].set_title('MEG', fontsize=14)
        gen_grid[1].text(-0.09, 1.0, 'B', transform=gen_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = gen_grid.cbar_axes[0].colorbar(im)
        gen_fig.suptitle('Gender {word} RDM Comparison\nScore: {score}'.format(word=word,
                                                                          score=best_gen_score),
                     fontsize=18)

        gen_fig.savefig(SAVE_FIG.format(fig_type='gen-rdm',
                                          word=word,
                                          win_len=win_len,
                                          ov=overlap,
                                          dist=dist,
                                          avgTm=run_Word_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

    score_fig.suptitle('Kendall Tau Scores over Time', fontsize=18)
    score_fig.savefig(SAVE_FIG.format(fig_type='score-overlay',
                                        word='both',
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_Word_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

    plt.show()