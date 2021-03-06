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

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_{fig_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_nd{num_draws}.pdf'

AGE = {'boy': 'young',
       'girl': 'young',
       'man': 'old',
       'woman': 'old',
       'kicked': 'none',
       'helped': 'none',
       'punched': 'none',
       'approached': 'none'
}

GEN = {'boy': 'male',
       'girl': 'female',
       'man': 'male',
       'woman': 'female',
       'kicked': 'none',
       'helped': 'none',
       'punched': 'none',
       'approached': 'none'
       }

PLOT_TITLE = {'det': 'Determiner',
              'noun2': 'Second Noun',
              'eos': 'Post Sentence',
              'eos-full': 'Post Sentence\nAll Sentence Lengths'}

TEXT_PAD_X = -0.08
TEXT_PAD_Y = 1.025

VMAX = {'cosine': 1.0, 'euclidean': 25.0}

def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    upper_tri_inds = np.triu_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[upper_tri_inds],rdm2[upper_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


def make_model_rdm(labels, dist):
    rdm = np.empty((len(labels), len(labels)))
    for i_lab, lab in enumerate(labels):
        for j_lab in range(i_lab, len(labels)):
            if dist != 'edit':
                if lab == labels[j_lab]:
                    rdm[i_lab, j_lab] = 0.0
                else:
                    rdm[i_lab, j_lab] = VMAX[dist]
            else:
                rdm[i_lab, j_lab] = edit_distance(lab, labels[j_lab])
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
    if word != 'det':
        age_labels = [AGE[lab] for lab in labels]
        gen_labels = [GEN[lab] for lab in labels]
        age_rdm = make_model_rdm(age_labels, dist)
        gen_rdm = make_model_rdm(gen_labels, dist)
    else:
        age_rdm = []
        gen_rdm = []

    voice_rdm = make_model_rdm(voice_labels, dist)
    word_rdm = make_model_rdm(labels, dist)
    string_rdm = make_model_rdm(labels, 'edit')
    return np.concatenate(subject_rdms, axis=0), word_rdm, string_rdm, voice_rdm, age_rdm, gen_rdm, time


def edit_distance(string1, string2):
    m=len(string1)+1
    n=len(string2)+1

    tbl = np.empty((m, n))
    for i in range(m):
        tbl[i,0]=float(i)
    for j in range(n):
        tbl[0,j]=float(j)
    for i in range(1, m):
        for j in range(1, n):
            if string1[i - 1] == string2[j - 1]:
                cost = 0.0
            else:
                cost= 1.0
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[m-1,n-1]


def noise_ceiling_krieg(subject_rdms):
    num_subjects = subject_rdms.shape[0]
    num_time = subject_rdms.shape[1]
    avg_rdm = np.squeeze(np.mean(subject_rdms, axis=0))
    noise_scores = np.empty((2, num_time, num_subjects))
    for i_sub in range(num_subjects):
        time_rdm = np.squeeze(subject_rdms[i_sub, ...])
        sub_inds = np.arange(num_subjects) != i_sub
        sub_less_avg = np.squeeze(np.mean(subject_rdms[sub_inds, ...], axis=0))
        for i_time in range(num_time):
            full_rdm = np.squeeze(avg_rdm[i_time, :, :])
            less_rdm = np.squeeze(sub_less_avg[i_time, :, :])
            rdm = np.squeeze(time_rdm[i_time, :, :])
            noise_scores[0, i_time, i_sub], _ = ktau_rdms(rdm, less_rdm)
            noise_scores[1, i_time, i_sub], _ = ktau_rdms(rdm, full_rdm)
    lower_bound = np.min(noise_scores[0, :, :], axis=2)
    upper_bound = np.max(noise_scores[1, :, :], axis=2)
    return lower_bound, upper_bound


def my_noise_ceiling(subject_rdms, num_draws):
    num_subjects = subject_rdms.shape[0]
    print(num_subjects)
    num_time = subject_rdms.shape[1]
    noise_scores = np.empty((num_draws, num_time))
    subject_list = range(num_subjects)
    for i_draw in range(num_draws):
        sample_1 = np.random.choice(subject_list, num_subjects/2)
        sample_2 = [sub for sub in subject_list if sub not in sample_1]
        sample_1_avg = np.mean(subject_rdms[sample_1, :, :, :], axis=0)
        sample_2_avg = np.mean(subject_rdms[sample_2, :, :, :], axis=0)
        for i_time in range(num_time):
            noise_scores[i_draw, i_time], _ = ktau_rdms(np.squeeze(sample_1_avg[i_time, ...]),
                                                        np.squeeze(sample_2_avg[i_time, ...]))
    lower_bound = np.min(noise_scores, axis=0)
    upper_bound = np.max(noise_scores, axis=0)
    med = np.median(noise_scores, axis=0)
    return lower_bound, med, upper_bound



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int, default=3)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='True', choices=['True', 'False'])
    parser.add_argument('--num_draws', type=int, default=100)

    args = parser.parse_args()

    experiment = args.experiment
    win_len = args.win_len
    overlap = args.overlap
    dist = args.dist
    doTimeAvg = run_Word_RSA.str_to_bool(args.doTimeAvg)
    num_draws = args.num_draws

    score_fig = plt.figure(figsize=(18, 9))
    score_grid = AxesGrid(score_fig, 111, nrows_ncols=(1, 4),
                        axes_pad=0.4, cbar_pad=0.4)
    colors = ['r', 'g', 'b', 'm', 'c']
    for i_word, word in enumerate(['noun2', 'det', 'eos-full', 'eos']):
        if word != 'det':
            subject_rdms, word_rdm, string_rdm, voice_rdm, age_rdm, gen_rdm, time = load_all_rdms(experiment,
                                                                                                  word,
                                                                                                  win_len,
                                                                                                  overlap,
                                                                                                  dist,
                                                                                                  run_Word_RSA.bool_to_str(doTimeAvg))
        else:
            subject_rdms, _, _, _, _, _, time = load_all_rdms(experiment,
                                                                word,
                                                                win_len,
                                                                overlap,
                                                                dist,
                                                                run_Word_RSA.bool_to_str(doTimeAvg))
        lower_bound, med, upper_bound = my_noise_ceiling(subject_rdms, num_draws)
        rdm = np.squeeze(np.mean(subject_rdms, axis=0))
        num_time = rdm.shape[0]

        time += win_len*0.002

        voice_scores_win = np.empty((num_time,))
        age_scores_win = np.empty((num_time,))
        gen_scores_win = np.empty((num_time,))
        word_scores_win = np.empty((num_time,))
        string_scores_win = np.empty((num_time,))
        for i_win in range(num_time):
            voice_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), voice_rdm)
            age_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), age_rdm)
            gen_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), gen_rdm)
            word_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), word_rdm)
            string_scores_win[i_win], _ = ktau_rdms(np.squeeze(rdm[i_win, :, :]), string_rdm)

        if i_word < 2:
            axis_ind = 1 - i_word
        else:
            axis_ind = i_word
        ax = score_grid[axis_ind]
        ax.plot(time, voice_scores_win, label='Voice', color=colors[0])
        ax.plot(time, word_scores_win, label='Word', color=colors[1])
        ax.plot(time, string_scores_win, label='Edit Distance', color=colors[2])
        if word != 'eos-full':
            ax.plot(time, age_scores_win, label='Age', color=colors[3])
            ax.plot(time, gen_scores_win, label='Gen', color=colors[4])
        ax.plot(time, upper_bound, label='Noise Ceiling UB', linestyle='-.', color='0.5')
        ax.plot(time, med, label='Noise Ceiling Med', linestyle='-', color='0.5')
        ax.plot(time, lower_bound, label='Noise Ceiling LB', linestyle='--', color='0.5')

        if axis_ind == 3:
            ax.legend(loc=1)

        ax.set_title('{word}'.format(word=PLOT_TITLE[word]), fontsize=14)
        ax.set_xlabel('Time (s)')
        if axis_ind == 0:
            ax.set_ylabel('Kendall Tau Correlation')

        ax.set_ylim([0.0, 0.7])
        ax.set_xlim([np. min(time), np.max(time)])
        ax.text(TEXT_PAD_X, TEXT_PAD_Y, string.ascii_uppercase[axis_ind], transform=ax.transAxes,
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
        best_word_win = np.argmax(word_scores_win)
        print('Best Word Correlation occurs at {}'.format(time[best_word_win]))
        best_word_score = word_scores_win[best_word_win]
        best_string_win = np.argmax(string_scores_win)
        print('Best Edit Distance Correlation occurs at {}'.format(time[best_string_win]))
        best_string_score = string_scores_win[best_string_win]

        voice_fig = plt.figure(figsize=(14, 7))
        voice_grid = AxesGrid(voice_fig, 111, nrows_ncols=(1, 2),
                                  axes_pad=0.4, cbar_mode='single', cbar_location='right',
                                  cbar_pad=0.4)
        voice_grid[0].imshow(voice_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        voice_grid[0].set_title('Model', fontsize=14)
        voice_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=voice_grid[0].transAxes,
                                            size=20, weight='bold')
        im = voice_grid[1].imshow(np.squeeze(rdm[best_voice_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # print(np.squeeze(rdm[best_voice_win, ...]))
        voice_grid[1].set_title('MEG', fontsize=14)
        voice_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=voice_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = voice_grid.cbar_axes[0].colorbar(im)
        voice_fig.suptitle('Voice {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
                                                                          score=best_voice_score),
                     fontsize=18)

        voice_fig.savefig(SAVE_FIG.format(fig_type='voice-rdm',
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_Word_RSA.bool_to_str(doTimeAvg),
                                          num_draws=num_draws), bbox_inches='tight')

        age_fig = plt.figure(figsize=(14, 7))
        age_grid = AxesGrid(age_fig, 111, nrows_ncols=(1, 2),
                              axes_pad=0.4, cbar_mode='single', cbar_location='right',
                              cbar_pad=0.4)
        age_grid[0].imshow(age_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        age_grid[0].set_title('Model', fontsize=14)
        age_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=age_grid[0].transAxes,
                           size=20, weight='bold')
        im = age_grid[1].imshow(np.squeeze(rdm[best_age_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # print(np.squeeze(rdm[best_age_win, ...]))
        age_grid[1].set_title('MEG', fontsize=14)
        age_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=age_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = age_grid.cbar_axes[0].colorbar(im)
        age_fig.suptitle('Age {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
                                                                          score=best_age_score),
                     fontsize=18)

        age_fig.savefig(SAVE_FIG.format(fig_type='age-rdm',
                                          word=word,
                                          win_len=win_len,
                                          ov=overlap,
                                          dist=dist,
                                          avgTm=run_Word_RSA.bool_to_str(doTimeAvg),
                                        num_draws=num_draws), bbox_inches='tight')

        gen_fig = plt.figure(figsize=(14, 7))
        gen_grid = AxesGrid(gen_fig, 111, nrows_ncols=(1, 2),
                              axes_pad=0.4, cbar_mode='single', cbar_location='right',
                              cbar_pad=0.4)
        gen_grid[0].imshow(gen_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        gen_grid[0].set_title('Model', fontsize=14)
        gen_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=gen_grid[0].transAxes,
                           size=20, weight='bold')
        im = gen_grid[1].imshow(np.squeeze(rdm[best_gen_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # print(np.squeeze(rdm[best_gen_win, ...]))
        gen_grid[1].set_title('MEG', fontsize=14)
        gen_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=gen_grid[1].transAxes,
                           size=20, weight='bold')
        cbar = gen_grid.cbar_axes[0].colorbar(im)
        gen_fig.suptitle('Gender {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
                                                                          score=best_gen_score),
                     fontsize=18)

        gen_fig.savefig(SAVE_FIG.format(fig_type='gen-rdm',
                                          word=word,
                                          win_len=win_len,
                                          ov=overlap,
                                          dist=dist,
                                          avgTm=run_Word_RSA.bool_to_str(doTimeAvg),
                                        num_draws=num_draws), bbox_inches='tight')

        word_fig = plt.figure(figsize=(14, 7))
        word_grid = AxesGrid(word_fig, 111, nrows_ncols=(1, 2),
                            axes_pad=0.4, cbar_mode='single', cbar_location='right',
                            cbar_pad=0.4)
        word_grid[0].imshow(word_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        word_grid[0].set_title('Model', fontsize=14)
        word_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=word_grid[0].transAxes,
                         size=20, weight='bold')
        im = word_grid[1].imshow(np.squeeze(rdm[best_word_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # print(np.squeeze(rdm[best_word_win, ...]))
        word_grid[1].set_title('MEG', fontsize=14)
        word_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=word_grid[1].transAxes,
                         size=20, weight='bold')
        cbar = word_grid.cbar_axes[0].colorbar(im)
        word_fig.suptitle('Word ID {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
                                                                               score=best_word_score),
                         fontsize=18)

        word_fig.savefig(SAVE_FIG.format(fig_type='word-rdm',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_Word_RSA.bool_to_str(doTimeAvg),
                                         num_draws=num_draws), bbox_inches='tight')

        string_fig = plt.figure(figsize=(14, 7))
        string_grid = AxesGrid(string_fig, 111, nrows_ncols=(1, 2),
                             axes_pad=0.4, cbar_mode='single', cbar_location='right',
                             cbar_pad=0.4)
        string_grid[0].imshow(string_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        string_grid[0].set_title('Model', fontsize=14)
        string_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=string_grid[0].transAxes,
                          size=20, weight='bold')
        im = string_grid[1].imshow(np.squeeze(rdm[best_string_win, ...]), interpolation='nearest', vmin=0.0,
                                 vmax=VMAX[dist])
        # print(np.squeeze(rdm[best_string_win, ...]))
        string_grid[1].set_title('MEG', fontsize=14)
        string_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=string_grid[1].transAxes,
                          size=20, weight='bold')
        cbar = string_grid.cbar_axes[0].colorbar(im)
        string_fig.suptitle('Edit Distance {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
                                                                                 score=best_string_score),
                          fontsize=18)

        string_fig.savefig(SAVE_FIG.format(fig_type='string-rdm',
                                         word=word,
                                         win_len=win_len,
                                         ov=overlap,
                                         dist=dist,
                                         avgTm=run_Word_RSA.bool_to_str(doTimeAvg),
                                        num_draws=num_draws), bbox_inches='tight')

    score_fig.suptitle('Kendall Tau Scores over Time', fontsize=18)
    score_fig.savefig(SAVE_FIG.format(fig_type='score-overlay',
                                        word='both',
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_Word_RSA.bool_to_str(doTimeAvg),
                                        num_draws=num_draws), bbox_inches='tight')

    plt.show()