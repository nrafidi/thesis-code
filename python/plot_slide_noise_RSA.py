import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import os
from scipy.stats import spearmanr, kendalltau
import run_slide_noise_RSA
from mpl_toolkits.axes_grid1 import AxesGrid
import string

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_{fig_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_{full_str}_split.pdf'
SAVE_SCORES = '/share/volume0/nrafidi/RSA_scores/{exp}/RSA_{score_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_{full_str}_split.npz'

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
              'last-full': 'Last Word',
              'eos': 'Post Sentence',
              'eos-full': 'Post Sentence All'}

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
    top_dir = run_slide_noise_RSA.TOP_DIR.format(exp=experiment)
    subject_val_rdms = []
    subject_test_rdms = []
    for i_subject, subject in enumerate(run_slide_noise_RSA.VALID_SUBS[experiment]):
        save_dir = run_slide_noise_RSA.SAVE_DIR.format(top_dir=top_dir, sub=subject)
        val_rdms = []
        test_rdms = []
        for draw in range(126):
            fname = run_slide_noise_RSA.SAVE_FILE.format(dir=save_dir,
                                                          sub=subject,
                                                          word=word,
                                                          win_len=win_len,
                                                          ov=overlap,
                                                          dist=dist,
                                                          avgTm=avgTm,
                                                         draw=draw)
            result = np.load(fname + '.npz')
            new_labels = result['labels']
            new_voice_labels = result['voice_labels']
            new_time = result['time'][result['win_starts']]
            if i_subject == 0 and draw == 0:
                labels = new_labels
                voice_labels = new_voice_labels
                time = new_time

            assert np.all(np.array(new_labels) == np.array(labels))
            assert np.all(np.array(new_voice_labels) == np.array(voice_labels))
            assert np.all(np.array(new_time) == np.array(time))

            val_rdms.append(result['val_rdm'][None, ...])
            test_rdms.append(result['test_rdm'][None, ...])
        val_rdm = np.concatenate(val_rdms, axis=0)
        test_rdm = np.concatenate(test_rdms, axis=0)
        subject_val_rdms.append(val_rdm[None, ...])
        subject_test_rdms.append(test_rdm[None, ...])
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
    return np.concatenate(subject_val_rdms, axis=0), np.concatenate(subject_test_rdms, axis=0), word_rdm, string_rdm, voice_rdm, age_rdm, gen_rdm, time


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


# assuming draw x time x stim x stim
def score_rdms(val_rdms, test_rdms):
    num_draws = test_rdms.shape[0]
    num_time = test_rdms.shape[1]
    scores = np.empty((num_draws, num_time))
    for i_draw in range(num_draws):
        for i_time in range(num_time):
            if len(val_rdms.shape) == 4:
                val = np.squeeze(val_rdms[i_draw, i_time, ...])
            else:
                val = val_rdms
            scores[i_draw, i_time], _ = ktau_rdms(val,
                                                  np.squeeze(test_rdms[i_draw, i_time, ...]))
    return scores



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int, default=3)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='True', choices=['True', 'False'])
    parser.add_argument('--plotFullSen', action='store_true')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    experiment = args.experiment
    win_len = args.win_len
    overlap = args.overlap
    dist = args.dist
    doTimeAvg = run_slide_noise_RSA.str_to_bool(args.doTimeAvg)
    plotFullSen = args.plotFullSen
    force = args.force

    if plotFullSen:
        word_list = ['last-full', 'eos-full']
        full_str = 'all'
    else:
        word_list = ['noun2', 'det', 'eos']
        full_str = 'long'

    score_fig = plt.figure(figsize=(18, 9))
    score_grid = AxesGrid(score_fig, 111, nrows_ncols=(1, len(word_list)),
                        axes_pad=0.4, cbar_pad=0.4)
    colors = ['r', 'g', 'b', 'm', 'c']
    for i_word, word in enumerate(word_list):
        if word != 'det':
            sub_val_rdms, sub_test_rdms, word_rdm, string_rdm, voice_rdm, age_rdm, gen_rdm, time = load_all_rdms(experiment,
                                                                                                                  word,
                                                                                                                  win_len,
                                                                                                                  overlap,
                                                                                                                  dist,
                                                                                                                  run_slide_noise_RSA.bool_to_str(doTimeAvg))
        else:
            sub_val_rdms, sub_test_rdms, _, _, _, _, _, time = load_all_rdms(experiment,
                                                                            word,
                                                                            win_len,
                                                                            overlap,
                                                                            dist,
                                                                            run_slide_noise_RSA.bool_to_str(doTimeAvg))
        val_rdms = np.squeeze(np.mean(sub_val_rdms, axis=0))
        test_rdms = np.squeeze(np.mean(sub_test_rdms, axis=0))
        rdm = np.squeeze(np.mean(test_rdms, axis=0))

        noise_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='noise',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(noise_file) and not force:
            noise_ceiling = score_rdms(val_rdms, test_rdms)
            np.savez_compressed(noise_file, noise_ceiling=noise_ceiling)
        else:
            result = np.load(noise_file)
            noise_ceiling = result['noise_ceiling']
        mean_noise = np.squeeze(np.mean(noise_ceiling, axis=0))
        std_noise = np.squeeze(np.std(noise_ceiling, axis=0))

        num_sub_test = 5

        fig, ax = plt.subplots()
        for i_sub in range(num_sub_test):
            sub_noise_score = score_rdms(np.squeeze(sub_val_rdms[i_sub, ...]),
                                                       np.squeeze(sub_test_rdms[i_sub, ...]))
            mean_sub_score = np.squeeze(np.mean(sub_noise_score, axis=0))
            print(mean_sub_score.shape)
            std_sub_score = np.squeeze(np.std(sub_noise_score, axis=0))
            ax.plot(time, mean_sub_score)
            ax.fill_between(time, mean_sub_score - std_sub_score, mean_sub_score + std_sub_score, alpha = 0.2)
        ax.set_title('{} Subject Noise Ceilings'.format(num_sub_test))
        # plt.show()





        num_time = test_rdms.shape[1]

        time += win_len*0.002

        voice_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='voice',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(voice_file) and not force:
            voice_scores = score_rdms(voice_rdm, test_rdms)
            np.savez_compressed(voice_file, voice_scores=voice_scores)
        else:
            result = np.load(voice_file)
            voice_scores = result['voice_scores']
        mean_voice = np.squeeze(np.mean(voice_scores, axis=0))
        std_voice = np.squeeze(np.std(voice_scores, axis=0))
        # age_scores = score_rdms(age_rdm, test_rdms)
        # mean_age = np.squeeze(np.mean(age_scores, axis=0))
        # std_age = np.squeeze(np.std(age_scores, axis=0))
        # gen_scores = score_rdms(gen_rdm, test_rdms)
        # mean_gen = np.squeeze(np.mean(gen_scores, axis=0))
        # std_gen = np.squeeze(np.std(gen_scores, axis=0))
        word_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='word',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(word_file) and not force:
            word_scores = score_rdms(word_rdm, test_rdms)
            np.savez_compressed(word_file, word_scores=word_scores)
        else:
            result = np.load(word_file)
            word_scores = result['word_scores']
        mean_word = np.squeeze(np.mean(word_scores, axis=0))
        std_word = np.squeeze(np.std(word_scores, axis=0))

        string_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='string',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(string_file) and not force:
            string_scores = score_rdms(string_rdm, test_rdms)
            np.savez_compressed(string_file, string_scores=string_scores)
        else:
            result = np.load(string_file)
            string_scores = result['string_scores']
        mean_string = np.squeeze(np.mean(string_scores, axis=0))
        std_string = np.squeeze(np.std(string_scores, axis=0))

        if i_word < 2 and not plotFullSen:
            axis_ind = 1 - i_word
        else:
            axis_ind = i_word
        ax = score_grid[axis_ind]
        ax.plot(time, mean_voice, label='Voice', color=colors[0])
        ax.fill_between(time, mean_voice - std_voice, mean_voice + std_voice,
                        facecolor=colors[0], alpha=0.5, edgecolor='w')
        ax.plot(time, mean_word, label='Word ID', color=colors[1])
        ax.fill_between(time, mean_word - std_word, mean_word + std_word,
                        facecolor=colors[1], alpha=0.5, edgecolor='w')
        ax.plot(time, mean_string, label='String Edit Distance', color=colors[2])
        ax.fill_between(time, mean_string - std_string, mean_string + std_string,
                        facecolor=colors[2], alpha=0.5, edgecolor='w')
        # if not plotFullSen:
        #     ax.plot(time, mean_age, label='Age', color=colors[3])
        #     ax.fill_between(time, mean_age - std_age, mean_age + std_age,
        #                     facecolor=colors[3], alpha=0.5, edgecolor='w')
        #     ax.plot(time, mean_gen, label='Gender', color=colors[4])
        #     ax.fill_between(time, mean_gen - std_gen, mean_gen + std_gen,
        #                     facecolor=colors[4], alpha=0.5, edgecolor='w')

        ax.plot(time, mean_noise, label='Noise Ceiling', linestyle='--', color='0.5')
        ax.fill_between(time, mean_noise - std_noise, mean_noise + std_noise,
                        facecolor='0.5', alpha=0.5, edgecolor='w')

        if axis_ind == len(word_list) - 1:
            ax.legend(loc=1)

        ax.set_title('{word}'.format(word=PLOT_TITLE[word]), fontsize=14)
        ax.set_xlabel('Time (s)')
        if axis_ind == 0:
            ax.set_ylabel('Kendall Tau Correlation')

        ax.set_ylim([0.0, 0.7])
        ax.set_xlim([np. min(time), np.max(time)])
        ax.text(TEXT_PAD_X, TEXT_PAD_Y, string.ascii_uppercase[axis_ind], transform=ax.transAxes,
                           size=20, weight='bold')

        # print(word)
        best_voice_win = np.argmax(mean_voice)
        print('Best Voice Correlation occurs at {}'.format(time[best_voice_win]))
        best_voice_score = mean_voice[best_voice_win]
        # best_age_win = np.argmax(mean_age)
        # print('Best Age Correlation occurs at {}'.format(time[best_age_win]))
        # best_age_score = mean_age[best_age_win]
        # best_gen_win = np.argmax(mean_gen)
        # print('Best Gen Correlation occurs at {}'.format(time[best_gen_win]))
        # best_gen_score = mean_gen[best_gen_win]
        best_word_win = np.argmax(mean_word)
        print('Best Word Correlation occurs at {}'.format(time[best_word_win]))
        best_word_score = mean_word[best_word_win]
        best_string_win = np.argmax(mean_string)
        print('Best Edit Distance Correlation occurs at {}'.format(time[best_string_win]))
        best_string_score = mean_string[best_string_win]

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
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str), bbox_inches='tight')

        # age_fig = plt.figure(figsize=(14, 7))
        # age_grid = AxesGrid(age_fig, 111, nrows_ncols=(1, 2),
        #                       axes_pad=0.4, cbar_mode='single', cbar_location='right',
        #                       cbar_pad=0.4)
        # age_grid[0].imshow(age_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # age_grid[0].set_title('Model', fontsize=14)
        # age_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=age_grid[0].transAxes,
        #                    size=20, weight='bold')
        # im = age_grid[1].imshow(np.squeeze(rdm[best_age_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # # print(np.squeeze(rdm[best_age_win, ...]))
        # age_grid[1].set_title('MEG', fontsize=14)
        # age_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=age_grid[1].transAxes,
        #                    size=20, weight='bold')
        # cbar = age_grid.cbar_axes[0].colorbar(im)
        # age_fig.suptitle('Age {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
        #                                                                   score=best_age_score),
        #              fontsize=18)
        #
        # age_fig.savefig(SAVE_FIG.format(fig_type='age-rdm',
        #                                   word=word,
        #                                   win_len=win_len,
        #                                   ov=overlap,
        #                                   dist=dist,
        #                                   avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')
        #
        # gen_fig = plt.figure(figsize=(14, 7))
        # gen_grid = AxesGrid(gen_fig, 111, nrows_ncols=(1, 2),
        #                       axes_pad=0.4, cbar_mode='single', cbar_location='right',
        #                       cbar_pad=0.4)
        # gen_grid[0].imshow(gen_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # gen_grid[0].set_title('Model', fontsize=14)
        # gen_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=gen_grid[0].transAxes,
        #                    size=20, weight='bold')
        # im = gen_grid[1].imshow(np.squeeze(rdm[best_gen_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # # print(np.squeeze(rdm[best_gen_win, ...]))
        # gen_grid[1].set_title('MEG', fontsize=14)
        # gen_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=gen_grid[1].transAxes,
        #                    size=20, weight='bold')
        # cbar = gen_grid.cbar_axes[0].colorbar(im)
        # gen_fig.suptitle('Gender {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
        #                                                                   score=best_gen_score),
        #              fontsize=18)
        #
        # gen_fig.savefig(SAVE_FIG.format(fig_type='gen-rdm',
        #                                   word=word,
        #                                   win_len=win_len,
        #                                   ov=overlap,
        #                                   dist=dist,
        #                                   avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

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
                                         full_str=full_str,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

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
                                         avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

    score_fig.suptitle('Kendall Tau Scores over Time', fontsize=18)
    score_fig.savefig(SAVE_FIG.format(fig_type='score-overlay',
                                        word='both',
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        full_str=full_str,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

    plt.show()