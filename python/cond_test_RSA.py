import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.stats import pearsonr
from scipy import spatial
import os
from scipy.stats import spearmanr, kendalltau
import run_slide_noise_RSA
from mpl_toolkits.axes_grid1 import AxesGrid
import string
from sklearn.linear_model import LinearRegression

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_{fig_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}'
SAVE_SCORES = '/share/volume0/nrafidi/RSA_scores/{exp}/RSA_{score_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}.npz'

Length = {'boy': 'noun',
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


TEXT_PAD_X = -0.125
TEXT_PAD_Y = 1.02

def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    upper_tri_inds = np.triu_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[upper_tri_inds],rdm2[upper_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue

def partial_ktau_rdms(X, Y, Z):
    rdmX = np.copy(X)
    rdmY = np.copy(Y)
    rdmZ = np.copy(Z)
    # Partial correlation between X and Y, conditioned on Z
    model_XZ = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    model_YZ = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

    model_XZ.fit(rdmZ, rdmX)
    model_YZ.fit(rdmZ, rdmY)

    residual_X = rdmX - model_XZ.predict(rdmZ)
    residual_Y = rdmY - model_YZ.predict(rdmZ)

    print(np.min(np.abs(residual_X)))
    print(np.min(np.abs(residual_Y)))

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0][0].imshow(rdmX, interpolation='nearest')
    axs[0][0].set_title('Original X')
    axs[0][1].imshow(residual_X, interpolation='nearest')
    axs[0][1].set_title('Residual X')
    axs[1][0].imshow(rdmY, interpolation='nearest')
    axs[1][0].set_title('Original Y')
    axs[1][1].imshow(residual_Y, interpolation='nearest')
    axs[1][1].set_title('Residual Y')


    meow, _ = ktau_rdms(residual_X, rdmZ)
    woof, _ = ktau_rdms(residual_Y, rdmZ)
    print('Residual corrs:')
    print(meow)
    print(woof)
    rdm_k_tau, rdm_k_tau_p = ktau_rdms(residual_X, residual_Y)
    return rdm_k_tau, rdm_k_tau_p, residual_X, residual_Y


def compute_partial(ktau_XY, ktau_XZ, ktau_YZ):
    num = ktau_XY - ktau_XZ*ktau_YZ
    print('ktau_XZ: {}'.format(ktau_XZ))
    print('ktau_YZ: {}'.format(ktau_YZ))
    print('Numerator: {}'.format(num))
    denom = np.sqrt(((1.0 - ktau_XZ)**2)*((1.0 - ktau_YZ)**2))
    print('Denomenator: {}'.format(denom))
    return num/denom


def alt_partial_corr_rdms(X, Y, Zs):
    if len(X.shape) == 2:
        X = spatial.distance.squareform(X, force='tovector', checks=False)
    if len(Y.shape) == 2:
        Y = spatial.distance.squareform(Y, force='tovector', checks=False)
    r_XY, _ = pearsonr(X, Y)
    print('Original XY: {}'.format(r_XY))
    for Z in Zs:
        if len(Z.shape) == 2:
            Z = spatial.distance.squareform(Z, force='tovector', checks=False)
        r_XZ, _ = pearsonr(X, Z)
        r_YZ, _ = pearsonr(Y, Z)
        r_XY = compute_partial(r_XY, r_XZ, r_YZ)
    print('Final XY: {}'.format(r_XY))
    return r_XY


def make_model_rdm(labels):
    rdm = np.empty((len(labels), len(labels)))
    for i_lab, lab in enumerate(labels):
        for j_lab in range(i_lab, len(labels)):
            if lab == labels[j_lab]:
                rdm[i_lab, j_lab] = 0.0
            else:
                rdm[i_lab, j_lab] = 1.0
            rdm[j_lab, i_lab] = rdm[i_lab, j_lab]
    return rdm


def make_syntax_rdm(len_labels, voice_labels):
    rdm = np.empty((len(len_labels), len(len_labels)))
    for i_lab, lab in enumerate(len_labels):
        voice_i = voice_labels[i_lab]
        for j_lab in range(i_lab, len(len_labels)):
            len_j = len_labels[j_lab]
            voice_j = voice_labels[j_lab]
            if voice_i != voice_j:
                rdm[i_lab, j_lab] = 1.0
            elif lab == len_j:
                rdm[i_lab, j_lab] = 0.0
            else:
                rdm[i_lab, j_lab] = 0.5
            rdm[j_lab, i_lab] = rdm[i_lab, j_lab]
    return rdm


def load_all_rdms(experiment, word, win_len, overlap, dist, avgTm):
    top_dir = run_slide_noise_RSA.TOP_DIR.format(exp=experiment)
    subject_val_rdms = []
    subject_test_rdms = []
    subject_total_rdms = []
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
            if draw == 0:
                subject_total_rdms.append(result['total_rdm'][None, ...])
        val_rdm = np.concatenate(val_rdms, axis=0)
        test_rdm = np.concatenate(test_rdms, axis=0)
        subject_val_rdms.append(val_rdm[None, ...])
        subject_test_rdms.append(test_rdm[None, ...])

    pos_labels = [Length[lab] for lab in labels]
    pos_rdm = make_model_rdm(pos_labels)
    len_labels = [LENGTH[voice_lab][word][pos_labels[i_lab]] for i_lab, voice_lab in enumerate(voice_labels)]
    syn_rdm = make_syntax_rdm(len_labels, voice_labels)
    voice_rdm = make_model_rdm(voice_labels)
    word_rdm = make_model_rdm(labels)

    return np.concatenate(subject_val_rdms, axis=0), np.concatenate(subject_test_rdms, axis=0), \
           np.concatenate(subject_total_rdms, axis=0), word_rdm, voice_rdm, pos_rdm, syn_rdm, time


# assuming draw x time x stim x stim
def score_rdms(val_rdms, test_rdms, cond_rdms=None):
    if len(val_rdms.shape) == 4:
        num_draws = val_rdms.shape[0]
        num_time = val_rdms.shape[1]
    elif len(test_rdms.shape) == 4:
        num_draws = test_rdms.shape[0]
        num_time = test_rdms.shape[1]
    else:
        num_draws = 1
        num_time = test_rdms.shape[0]
    num_time=2
    scores = np.empty((num_draws, num_time))
    for i_draw in range(num_draws):
        for i_time in range(num_time):
            if len(val_rdms.shape) == 4:
                val = np.squeeze(val_rdms[i_draw, i_time, ...])
            elif len(val_rdms.shape) == 3:
                val = np.squeeze(val_rdms[i_time, ...])
            else:
                val = val_rdms
            if len(test_rdms.shape) == 4:
                test = np.squeeze(test_rdms[i_draw, i_time, ...])
            elif len(test_rdms.shape) == 3:
                test = np.squeeze(test_rdms[i_time, ...])
            else:
                test = test_rdms
            if cond_rdms is None:
                scores[i_draw, i_time], _ = ktau_rdms(val, test)
            else:
                rdmX = val
                rdmY = test
                # scores[i_draw, i_time] = alt_partial_corr_rdms(rdmX, rdmY, cond_rdms)
                for cond_rdm in cond_rdms:
                    if len(cond_rdm.shape) == 4:
                        rdmZ = np.squeeze(cond_rdm[i_draw, i_time, ...])
                    elif len(cond_rdm.shape) == 3:
                        rdmZ = np.squeeze(cond_rdm[i_time, ...])
                    else:
                        rdmZ = cond_rdm
                    scores[i_draw, i_time], _, rdmX, rdmY = partial_ktau_rdms(rdmX, rdmY, rdmZ)

    return np.squeeze(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--win_len', type=int, default=25)
    parser.add_argument('--overlap', type=int, default=2)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='F', choices=['T', 'F'])
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    experiment = args.experiment
    overlap = args.overlap
    dist = args.dist

    force = args.force

    ticklabelsize = 16
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    win_len = args.win_len
    doTimeAvg = args.doTimeAvg
    word = 'eos-full'

    sub_val_rdms, sub_test_rdms, sub_total_rdms, word_rdm, voice_rdm, pos_rdm, syn_rdm, time = load_all_rdms(experiment,
                                                                                                              word,
                                                                                                              win_len,
                                                                                                              overlap,
                                                                                                              dist,
                                                                                                              doTimeAvg)

    syn_voice_scores, _ = ktau_rdms(voice_rdm, syn_rdm)
    print('Correlation between Voice and Syntax RDMs is: {}'.format(syn_voice_scores))

    rdm_fig = plt.figure(figsize=(12, 8))
    rdm_grid = AxesGrid(rdm_fig, 111, nrows_ncols=(1, 2),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True, aspect=True)

    rdms_to_plot = [syn_rdm, voice_rdm]
    titles_to_plot = ['Syntax', 'Voice']

    for i_rdm, rdm in enumerate(rdms_to_plot):
        im = rdm_grid[i_rdm].imshow(rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
        rdm_grid[i_rdm].set_title(titles_to_plot[i_rdm], fontsize=axistitlesize)
        rdm_grid[i_rdm].text(-0.15, 1.05, string.ascii_uppercase[i_rdm], transform=rdm_grid[i_rdm].transAxes,
                size=axislettersize, weight='bold')
    cbar = rdm_grid.cbar_axes[0].colorbar(im)
    rdm_fig.suptitle('RDM Comparison')
    rdm_fig.savefig(SAVE_FIG.format(fig_type='rdm-comp-voice',
                                      word=word,
                                      win_len=win_len,
                                      ov=overlap,
                                      dist=dist,
                                      avgTm=doTimeAvg) + '.png', bbox_inches='tight')
    rdm_fig.savefig(SAVE_FIG.format(fig_type='rdm-comp-voice',
                                    word=word,
                                    win_len=win_len,
                                    ov=overlap,
                                    dist=dist,
                                    avgTm=doTimeAvg) + '.pdf', bbox_inches='tight')


    total_avg_rdms = np.squeeze(np.mean(sub_total_rdms, axis=0))

    syn_rep_cond_file = SAVE_SCORES.format(exp=experiment,
                                           score_type='syn-rep-cond-voice',
                                           word=word,
                                           win_len=win_len,
                                           ov=overlap,
                                           dist=dist,
                                           avgTm=doTimeAvg)
    if os.path.isfile(syn_rep_cond_file) and not force:
        result = np.load(syn_rep_cond_file)
        syn_rep_cond_scores = result['scores']
    else:
        syn_rep_cond_scores = score_rdms(syn_rdm, total_avg_rdms, cond_rdms=[voice_rdm])
        np.savez_compressed(syn_rep_cond_file, scores=syn_rep_cond_scores)

    voice_rep_cond_file = SAVE_SCORES.format(exp=experiment,
                                           score_type='voice-rep-cond-syn',
                                           word=word,
                                           win_len=win_len,
                                           ov=overlap,
                                           dist=dist,
                                           avgTm=doTimeAvg)
    if os.path.isfile(voice_rep_cond_file) and not force:
        result = np.load(voice_rep_cond_file)
        voice_rep_cond_scores = result['scores']
    else:
        voice_rep_cond_scores = score_rdms(voice_rdm, total_avg_rdms, cond_rdms=[syn_rdm])
        np.savez_compressed(voice_rep_cond_file, scores=voice_rep_cond_scores)

    voice_rep_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='voice-rep',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvg)
    if os.path.isfile(voice_rep_file) and not force:
        result = np.load(voice_rep_file)
        voice_rep_scores = result['scores']
    else:
        voice_rep_scores = score_rdms(voice_rdm, total_avg_rdms)
        np.savez_compressed(voice_rep_file, scores=voice_rep_scores)



    syn_rep_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='syn-rep',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvg)
    if os.path.isfile(syn_rep_file) and not force:
        result = np.load(syn_rep_file)
        syn_rep_scores = result['scores']
    else:
        syn_rep_scores = score_rdms(syn_rdm, total_avg_rdms)
        np.savez_compressed(syn_rep_file, scores=syn_rep_scores)



    score_labels = ['Syntax 0 order', 'Syntax Conditional', 'Voice 0 order', 'Voice Conditional']
    scores_to_plot = [syn_rep_scores, syn_rep_cond_scores,
                      voice_rep_scores, voice_rep_cond_scores]
    cond_fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['r', 'g', 'm', 'b']

    for i_score, score in enumerate(scores_to_plot):
        # print(np.min(score))
        # print(np.max(score))
        ax.plot(score, color=colors[i_score], label=score_labels[i_score])

    ax.legend(fontsize=legendfontsize)
    ax.set_ylabel('Kendall tau', fontsize=axislabelsize)
    ax.set_ylim([-1.0, 1.0])
    ax.tick_params(labelsize=ticklabelsize)

    cond_fig.suptitle('Correlation Type Comparison Over Time', fontsize=suptitlesize)
    cond_fig.subplots_adjust(top=0.85)
    cond_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-cond-comp-voice-time',
                                      word=word,
                                      win_len=win_len,
                                      ov=overlap,
                                      dist=dist,
                                      avgTm=doTimeAvg) + '.png', bbox_inches='tight')

    cond_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-cond-comp-voice-time',
                                     word=word,
                                     win_len=win_len,
                                     ov=overlap,
                                     dist=dist,
                                     avgTm=doTimeAvg) + '.pdf', bbox_inches='tight')

    plt.show()