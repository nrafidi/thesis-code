import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.stats.mstats import normaltest
import os
from scipy.stats import spearmanr, kendalltau
import run_slide_noise_RSA
from mpl_toolkits.axes_grid1 import AxesGrid
import string
from sklearn.linear_model import LinearRegression

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_{fig_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_{full_str}_split.pdf'
SAVE_SCORES = '/share/volume0/nrafidi/RSA_scores/{exp}/RSA_{score_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_{full_str}_split.npz'

AGE = {'boy': 'young',
       'girl': 'young',
       'man': 'old',
       'woman': 'old',
       'kicked': 'none',
       'helped': 'none',
       'punched': 'none',
       'approached': 'none',
       'the': 'none',
       'by': 'none',
       'was': 'none'
}

GEN = {'boy': 'male',
       'girl': 'female',
       'man': 'male',
       'woman': 'female',
       'kicked': 'none',
       'helped': 'none',
       'punched': 'none',
       'approached': 'none',
       'the': 'none',
       'by': 'none',
       'was': 'none'
       }

POS = {'boy': 'noun',
       'girl': 'noun',
       'man': 'noun',
       'woman': 'noun',
       'kicked': 'verb',
       'helped': 'verb',
       'punched': 'verb',
       'approached': 'verb',
       'the': 'det',
       'by': 'help',
       'was': 'help'
       }

LENGTH = {'active': {'third-last-full': {'verb': 'long',
                                         'det': 'short'},
                     'second-last-full': {'det': 'long',
                                          'noun': 'short'},
                     'last-full': {'noun': 'long',
                                   'verb': 'short'},
                     'eos-full': {'noun': 'long',
                                  'verb': 'short'}
                     },
          'passive': {'third-last-full': {'help': 'long',
                                          'noun': 'short'},
                      'second-last-full': {'det': 'long',
                                           'help': 'short'},
                      'last-full': {'noun': 'long',
                                    'verb': 'short'},
                      'eos-full': {'noun': 'long',
                                   'verb': 'short'}
                      }
          }

PLOT_TITLE = {'det': 'Determiner',
              'noun2': 'Second Noun',
              'third-last': 'Third-to-Last Word',
              'third-last-full': 'Third-to-Last Word',
              'second-last-full': 'Second-to-Last Word',
              'last-full': 'Last Word',
              'eos': 'Post Sentence',
              'eos-full': 'Post Sentence All'}

TEXT_PAD_X = -0.1
TEXT_PAD_Y = 1.0225

VMAX = {'cosine': 1.0, 'euclidean': 25.0}

def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    upper_tri_inds = np.triu_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[upper_tri_inds],rdm2[upper_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


def partial_ktau_rdms(rdmX, rdmY, rdmZ):
    # Partial correlation between X and Y, conditioned on Z
    model_XZ = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    model_YZ = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

    model_XZ.fit(rdmZ, rdmX)
    model_YZ.fit(rdmZ, rdmY)

    residual_X = rdmX - model_XZ.predict(rdmZ)
    residual_Y = rdmY - model_YZ.predict(rdmZ)

    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # axs[0].imshow(residual_X, interpolation='nearest')
    # axs[1].imshow(residual_Y, interpolation='nearest')
    # plt.show()

    rdm_k_tau, rdm_k_tau_p = ktau_rdms(residual_X, residual_Y)
    return rdm_k_tau, rdm_k_tau_p



def make_model_rdm(labels, dist):
    rdm = np.empty((len(labels), len(labels)))
    for i_lab, lab in enumerate(labels):
        for j_lab in range(i_lab, len(labels)):
            if dist == 'edit':
                rdm[i_lab, j_lab] = edit_distance(lab, labels[j_lab])
            elif dist == 'len':
                rdm[i_lab, j_lab] = np.abs(len(lab) - len(labels[j_lab]))
            else:
                if lab == labels[j_lab]:
                    rdm[i_lab, j_lab] = 0.0
                else:
                    rdm[i_lab, j_lab] = VMAX[dist]
            rdm[j_lab, i_lab] = rdm[i_lab, j_lab]
    return rdm


def make_syntax_rdm(len_labels, voice_labels, dist):
    rdm = np.empty((len(len_labels), len(len_labels)))
    for i_lab, lab in enumerate(len_labels):
        voice_i = voice_labels[i_lab]
        for j_lab in range(i_lab, len(len_labels)):
            len_j = len_labels[j_lab]
            voice_j = voice_labels[j_lab]
            if voice_i != voice_j:
                rdm[i_lab, j_lab] = VMAX[dist]
            elif lab == len_j:
                rdm[i_lab, j_lab] = 0.0
            else:
                rdm[i_lab, j_lab] = 0.5*VMAX[dist]
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
        pos_labels = [POS[lab] for lab in labels]
        age_rdm = make_model_rdm(age_labels, dist)
        gen_rdm = make_model_rdm(gen_labels, dist)
        pos_rdm = make_model_rdm(pos_labels, dist)

        if 'full' in word:
            len_labels = [LENGTH[voice_lab][word][pos_labels[i_lab]] for i_lab, voice_lab in enumerate(voice_labels)]
            syn_rdm = make_syntax_rdm(len_labels, voice_labels, dist)
        else:
            syn_rdm = []
    else:
        age_rdm = []
        gen_rdm = []
        pos_rdm = []
        syn_rdm = []

    voice_rdm = make_model_rdm(voice_labels, dist)
    # print(labels)
    word_rdm = make_model_rdm(labels, dist)
    # print(word_rdm[:10, :10])
    # string_rdm = make_model_rdm(labels, 'edit')
    string_rdm = make_model_rdm(labels, 'len')
    # print(string_rdm[:10, :10])
    return np.concatenate(subject_val_rdms, axis=0), np.concatenate(subject_test_rdms, axis=0), word_rdm, string_rdm, \
           voice_rdm, age_rdm, gen_rdm, pos_rdm, syn_rdm, time


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
def score_rdms(val_rdms, test_rdms, cond_rdms=None):
    num_draws = test_rdms.shape[0]
    num_time = test_rdms.shape[1]
    scores = np.empty((num_draws, num_time))
    for i_draw in range(num_draws):
        for i_time in range(num_time):
            if len(val_rdms.shape) == 4:
                val = np.squeeze(val_rdms[i_draw, i_time, ...])
            else:
                val = val_rdms
            if cond_rdms is None:
                scores[i_draw, i_time], _ = ktau_rdms(val,
                                                      np.squeeze(test_rdms[i_draw, i_time, ...]))
            elif len(cond_rdms.shape) == 4:
                scores[i_draw, i_time], _ = partial_ktau_rdms(val,
                                                      np.squeeze(test_rdms[i_draw, i_time, ...]),
                                                              np.squeeze(cond_rdms[i_draw, i_time, ...]))
            else:
                scores[i_draw, i_time], _ = partial_ktau_rdms(val,
                                                              np.squeeze(test_rdms[i_draw, i_time, ...]),
                                                              cond_rdms)
    return scores



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int, default=3)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='True', choices=['True', 'False'])
    parser.add_argument('--plotFullSen', action='store_true')
    parser.add_argument('--cond', default='len', choices=['len', 'pos', 'word', 'None'])
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    experiment = args.experiment
    win_len = args.win_len
    overlap = args.overlap
    dist = args.dist
    doTimeAvg = run_slide_noise_RSA.str_to_bool(args.doTimeAvg)
    plotFullSen = args.plotFullSen
    cond = args.cond
    force = args.force

    if plotFullSen:
        word_list = ['third-last-full', 'second-last-full', 'last-full', 'eos-full']
        full_str = 'all'
    else:
        word_list = ['third-last', 'noun2', 'det', 'eos']
        full_str = 'long'

    score_fig = plt.figure(figsize=(25, 7))
    score_grid = AxesGrid(score_fig, 111, nrows_ncols=(1, len(word_list)),
                        axes_pad=0.4)
    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    for i_word, word in enumerate(word_list):
        if word != 'det':
            sub_val_rdms, sub_test_rdms, word_rdm, string_rdm, voice_rdm, age_rdm, gen_rdm, pos_rdm, syn_rdm, time = load_all_rdms(experiment,
                                                                                                                                  word,
                                                                                                                                  win_len,
                                                                                                                                  overlap,
                                                                                                                                  dist,
                                                                                                                                  run_slide_noise_RSA.bool_to_str(doTimeAvg))
        else:
            sub_val_rdms, sub_test_rdms, _, _, _, _, _, _, _, time = load_all_rdms(experiment,
                                                                                    word,
                                                                                    win_len,
                                                                                    overlap,
                                                                                    dist,
                                                                                    run_slide_noise_RSA.bool_to_str(doTimeAvg))
        if cond == 'len':
            rdm_Z_pos = string_rdm
            rdm_Z_word = string_rdm
            rdm_Z_syn = string_rdm
            rdm_Z_string = None
        elif cond == 'pos':
            rdm_Z_pos = None
            rdm_Z_word = pos_rdm
            rdm_Z_syn = pos_rdm
            rdm_Z_string = pos_rdm
        elif cond == 'word':
            rdm_Z_pos = word_rdm
            rdm_Z_word = None
            rdm_Z_syn = word_rdm
            rdm_Z_string = word_rdm
        else:
            rdm_Z_pos = None
            rdm_Z_word = None
            rdm_Z_syn = None
            rdm_Z_string = None

        val_rdms = np.squeeze(np.mean(sub_val_rdms, axis=0))
        test_rdms = np.squeeze(np.mean(sub_test_rdms, axis=0))
        of_rdms = (val_rdms + test_rdms)/2.0
        rdm = np.squeeze(np.mean(test_rdms, axis=0))

        remainders = np.mean(val_rdms - test_rdms, axis=0)
        remainders = remainders.flatten()

        # fig, ax = plt.subplots()
        # ax.hist(remainders)

        k2, p = normaltest(remainders)
        print('P value is : {}'.format(p))

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
            result = np.load(noise_file)
            noise_ceiling = result['noise_ceiling']
        else:
            noise_ceiling = score_rdms(val_rdms, test_rdms)
            np.savez_compressed(noise_file, noise_ceiling=noise_ceiling)

        noise_ub_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='noise-ub',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(noise_ub_file) and not force:
            result = np.load(noise_ub_file)
            noise_ub_ceiling = result['noise_ub_ceiling']
        else:
            noise_ub_ceiling = score_rdms(val_rdms, of_rdms)
            np.savez_compressed(noise_ub_file, noise_ub_ceiling=noise_ub_ceiling)

        mean_noise = np.squeeze(np.mean(noise_ceiling, axis=0))
        std_noise = np.squeeze(np.std(noise_ceiling, axis=0))

        mean_noise_ub = np.squeeze(np.mean(noise_ub_ceiling, axis=0))
        std_noise_ub = np.squeeze(np.std(noise_ub_ceiling, axis=0))

        num_time = test_rdms.shape[1]

        time += win_len*0.002

        voice_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='voice-ub',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(voice_file) and not force:
            result = np.load(voice_file)
            voice_scores = result['voice_scores']
        else:
            voice_scores = score_rdms(voice_rdm, of_rdms)
            np.savez_compressed(voice_file, voice_scores=voice_scores)
        mean_voice = np.squeeze(np.mean(voice_scores, axis=0))
        std_voice = np.squeeze(np.std(voice_scores, axis=0))

        if plotFullSen:
            gen_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='gen-ub',
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )
            if os.path.isfile(gen_file) and not force:
                result = np.load(gen_file)
                gen_scores = result['gen_scores']
            else:
                gen_scores = score_rdms(gen_rdm, of_rdms)
                np.savez_compressed(gen_file, gen_scores=gen_scores)
            mean_gen = np.squeeze(np.mean(gen_scores, axis=0))
            std_gen = np.squeeze(np.std(gen_scores, axis=0))

            pos_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='pos-ub-cond-{}'.format(cond),
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )
            if os.path.isfile(pos_file) and not force:
                result = np.load(pos_file)
                pos_scores = result['pos_scores']
            else:
                pos_scores = score_rdms(pos_rdm, of_rdms, rdm_Z_pos)
                np.savez_compressed(pos_file, pos_scores=pos_scores)
            mean_pos = np.squeeze(np.mean(pos_scores, axis=0))
            std_pos = np.squeeze(np.std(pos_scores, axis=0))

            syn_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='syn-ub-cond-{}'.format(cond),
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )
            if os.path.isfile(syn_file) and not force:
                result = np.load(syn_file)
                syn_scores = result['syn_scores']
            else:
                syn_scores = score_rdms(syn_rdm, of_rdms, rdm_Z_syn)
                np.savez_compressed(syn_file, syn_scores=syn_scores)
            mean_syn = np.squeeze(np.mean(syn_scores, axis=0))
            std_syn = np.squeeze(np.std(syn_scores, axis=0))
        
        
        
        word_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='word-ub-cond-{}'.format(cond),
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(word_file) and not force:
            result = np.load(word_file)
            word_scores = result['word_scores']
        else:
            word_scores = score_rdms(word_rdm, of_rdms, rdm_Z_word)
            np.savez_compressed(word_file, word_scores=word_scores)
        mean_word = np.squeeze(np.mean(word_scores, axis=0))
        std_word = np.squeeze(np.std(word_scores, axis=0))


        string_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='string-len-ub-cond-{}'.format(cond),
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                        full_str=full_str
                                        )
        if os.path.isfile(string_file) and not force:
            result = np.load(string_file)
            string_scores = result['string_scores']
        else:
            string_scores = score_rdms(string_rdm, of_rdms, rdm_Z_string)
            np.savez_compressed(string_file, string_scores=string_scores)
        mean_string = np.squeeze(np.mean(string_scores, axis=0))
        std_string = np.squeeze(np.std(string_scores, axis=0))

        if word == 'det':
            axis_ind = 1
        elif word == 'noun2':
            axis_ind = 2
        else:
            axis_ind = i_word
        ax = score_grid[axis_ind]

        ax.plot(time, mean_word, label='Word ID', color=colors[0])
        ax.fill_between(time, mean_word - std_word, mean_word + std_word,
                        facecolor=colors[0], alpha=0.5, edgecolor='w')
        # ax.plot(time, mean_word, label='Word ID UB', linestyle='--', color=colors[0])
        # ax.fill_between(time, mean_word_ub - std_word_ub, mean_word_ub + std_word_ub,
        #                 facecolor=colors[0], alpha=0.5, edgecolor='w')
        ax.plot(time, mean_string, label='Word Length', color=colors[1])
        ax.fill_between(time, mean_string - std_string, mean_string + std_string,
                        facecolor=colors[1], alpha=0.5, edgecolor='w')
        if plotFullSen:
            # ax.plot(time, mean_gen, label='Gender', color=colors[3])
            # ax.fill_between(time, mean_gen - std_gen, mean_gen + std_gen,
            #                 facecolor=colors[3], alpha=0.5, edgecolor='w')
            ax.plot(time, mean_pos, label='POS', color=colors[2])
            ax.fill_between(time, mean_pos - std_pos, mean_pos + std_pos,
                            facecolor=colors[2], alpha=0.5, edgecolor='w')
            ax.plot(time, mean_syn, label='Syntax', color=colors[3])
            ax.fill_between(time, mean_syn - std_syn, mean_syn + std_syn,
                            facecolor=colors[3], alpha=0.5, edgecolor='w')
        else:
            ax.plot(time, mean_voice, label='Voice', color=colors[2])
            ax.fill_between(time, mean_voice - std_voice, mean_voice + std_voice,
                            facecolor=colors[2], alpha=0.5, edgecolor='w')


        ax.plot(time, mean_noise, label='Noise LB', linestyle='--', color='0.5')
        ax.fill_between(time, mean_noise - std_noise, mean_noise + std_noise,
                        facecolor='0.5', alpha=0.5, edgecolor='w')
        ax.plot(time, mean_noise_ub, label='Noise UB', linestyle=':', color='0.5')
        ax.fill_between(time, mean_noise_ub - std_noise_ub, mean_noise_ub + std_noise_ub,
                        facecolor='0.5', alpha=0.5, edgecolor='w')

        if axis_ind == len(word_list) - 1:
            ax.legend(loc=1, ncol=2)

        ax.set_title('{word}'.format(word=PLOT_TITLE[word]), fontsize=14)
        # ax.set_xlabel('Time (s)')
        if axis_ind == 0:
            ax.set_ylabel('Kendall Tau Correlation', fontsize=14)

        ax.set_ylim([-0.1, 1.0])
        ax.set_xlim([np. min(time), np.max(time)])
        # ax.set_xticks(range(0, len(time), 5))
        ax.set_xticks(time[::20])
        ax.text(TEXT_PAD_X, TEXT_PAD_Y, string.ascii_uppercase[axis_ind], transform=ax.transAxes,
                           size=20, weight='bold')

        # # print(word)
        # best_voice_win = np.argmax(mean_voice)
        # print('Best Voice Correlation occurs at {}'.format(time[best_voice_win]))
        # best_voice_score = mean_voice[best_voice_win]
        # # best_age_win = np.argmax(mean_age)
        # # print('Best Age Correlation occurs at {}'.format(time[best_age_win]))
        # # best_age_score = mean_age[best_age_win]
        # # best_gen_win = np.argmax(mean_gen)
        # # print('Best Gen Correlation occurs at {}'.format(time[best_gen_win]))
        # # best_gen_score = mean_gen[best_gen_win]
        # best_word_win = np.argmax(mean_word)
        # print('Best Word Correlation occurs at {}'.format(time[best_word_win]))
        # best_word_score = mean_word[best_word_win]
        # best_string_win = np.argmax(mean_string)
        # print('Best Length Difference Correlation occurs at {}'.format(time[best_string_win]))
        # best_string_score = mean_string[best_string_win]

        # voice_fig = plt.figure(figsize=(14, 7))
        # voice_grid = AxesGrid(voice_fig, 111, nrows_ncols=(1, 2),
        #                           axes_pad=0.4, cbar_mode='single', cbar_location='right',
        #                           cbar_pad=0.4)
        # voice_grid[0].imshow(voice_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # voice_grid[0].set_title('Model', fontsize=14)
        # voice_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=voice_grid[0].transAxes,
        #                                     size=20, weight='bold')
        # im = voice_grid[1].imshow(np.squeeze(rdm[best_voice_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # # print(np.squeeze(rdm[best_voice_win, ...]))
        # voice_grid[1].set_title('MEG', fontsize=14)
        # voice_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=voice_grid[1].transAxes,
        #                    size=20, weight='bold')
        # cbar = voice_grid.cbar_axes[0].colorbar(im)
        # voice_fig.suptitle('Voice {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
        #                                                                   score=best_voice_score),
        #              fontsize=18)
        #
        # voice_fig.savefig(SAVE_FIG.format(fig_type='voice-rdm',
        #                                     word=word,
        #                                     win_len=win_len,
        #                                     ov=overlap,
        #                                     dist=dist,
        #                                     avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
        #                                     full_str=full_str), bbox_inches='tight')
        #
        # # age_fig = plt.figure(figsize=(14, 7))
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

        # word_fig = plt.figure(figsize=(14, 7))
        # word_grid = AxesGrid(word_fig, 111, nrows_ncols=(1, 2),
        #                     axes_pad=0.4, cbar_mode='single', cbar_location='right',
        #                     cbar_pad=0.4)
        # word_grid[0].imshow(word_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # word_grid[0].set_title('Model', fontsize=14)
        # word_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=word_grid[0].transAxes,
        #                  size=20, weight='bold')
        # im = word_grid[1].imshow(np.squeeze(rdm[best_word_win, ...]), interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # # print(np.squeeze(rdm[best_word_win, ...]))
        # word_grid[1].set_title('MEG', fontsize=14)
        # word_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=word_grid[1].transAxes,
        #                  size=20, weight='bold')
        # cbar = word_grid.cbar_axes[0].colorbar(im)
        # word_fig.suptitle('Word ID {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
        #                                                                        score=best_word_score),
        #                  fontsize=18)
        #
        # word_fig.savefig(SAVE_FIG.format(fig_type='word-rdm',
        #                                 word=word,
        #                                 win_len=win_len,
        #                                 ov=overlap,
        #                                 dist=dist,
        #                                  full_str=full_str,
        #                                 avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')
        #
        # string_fig = plt.figure(figsize=(14, 7))
        # string_grid = AxesGrid(string_fig, 111, nrows_ncols=(1, 2),
        #                      axes_pad=0.4, cbar_mode='single', cbar_location='right',
        #                      cbar_pad=0.4)
        # string_grid[0].imshow(string_rdm, interpolation='nearest', vmin=0.0, vmax=VMAX[dist])
        # string_grid[0].set_title('Model', fontsize=14)
        # string_grid[0].text(TEXT_PAD_X, TEXT_PAD_Y, 'A', transform=string_grid[0].transAxes,
        #                   size=20, weight='bold')
        # im = string_grid[1].imshow(np.squeeze(rdm[best_string_win, ...]), interpolation='nearest', vmin=0.0,
        #                          vmax=VMAX[dist])
        # # print(np.squeeze(rdm[best_string_win, ...]))
        # string_grid[1].set_title('MEG', fontsize=14)
        # string_grid[1].text(TEXT_PAD_X, TEXT_PAD_Y, 'B', transform=string_grid[1].transAxes,
        #                   size=20, weight='bold')
        # cbar = string_grid.cbar_axes[0].colorbar(im)
        # string_fig.suptitle('String Length Difference {word} RDM Comparison\nScore: {score}'.format(word=PLOT_TITLE[word],
        #                                                                          score=best_string_score),
        #                   fontsize=18)
        #
        # string_fig.savefig(SAVE_FIG.format(fig_type='string-rdm',
        #                                  word=word,
        #                                  win_len=win_len,
        #                                  ov=overlap,
        #                                  dist=dist,
        #                                    full_str=full_str,
        #                                  avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

    score_fig.suptitle('Kendall Tau Scores over Time', fontsize=18)

    score_fig.text(0.5, 0.04, 'Time Relative to Last Word Onset (s)', ha='center', fontsize=14)
    score_fig.savefig(SAVE_FIG.format(fig_type='score-overlay',
                                        word='both',
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        full_str=full_str,
                                        avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

    plt.show()