import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spacial.distance import squareform, pdist
import os
from scipy.stats import spearmanr, kendalltau
import run_slide_noise_RSA
from mpl_toolkits.axes_grid1 import AxesGrid
import string
from sklearn.linear_model import LinearRegression
from syntax_vs_semantics import load_data


MODEL_PATH = '/share/volume0/RNNG/semantic_models/glove/sentence_similarity/direct_sentence_distance/{experiment}_{model}_glove_rdm.npz'

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


NUM_LENGTH = {'active': {'third-last-full': {'verb': 5,
                                         'det': 3},
                     'second-last-full': {'det': 5,
                                          'noun': 3},
                     'last-full': {'noun': 5,
                                   'verb': 3},
                     'eos-full': {'noun': 5,
                                  'verb': 3}
                     },
          'passive': {'third-last-full': {'help': 7,
                                          'noun': 4},
                      'second-last-full': {'det': 7,
                                           'help': 4},
                      'last-full': {'noun':7,
                                    'verb': 4},
                      'eos-full': {'noun': 7,
                                   'verb': 4}
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

def partial_ktau_rdms(rdmX, rdmY, rdmZ):
    # Partial correlation between X and Y, conditioned on Z
    model_XZ = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    model_YZ = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

    model_XZ.fit(rdmZ, rdmX)
    model_YZ.fit(rdmZ, rdmY)

    residual_X = rdmX - model_XZ.predict(rdmZ)
    residual_Y = rdmY - model_YZ.predict(rdmZ)

    rdm_k_tau, rdm_k_tau_p = ktau_rdms(residual_X, residual_Y)
    return rdm_k_tau, rdm_k_tau_p, residual_X, residual_Y


def load_bow(experiment, distance='cosine'):
    model_embeddings = np.load('/share/volume0/RNNG/semantic_models/embeddings_dict.npz').item()['glove']

    from itertools import groupby
    structured_stimuli = list()
    for _, sentence_usis in groupby(
            load_data.filtered_query(experiment, filter_sets=[[]])[0], lambda usi: usi[1]['sentence_id']):
        structured_stimuli.append(list(sentence_usis))
    structured_stimuli = sorted(
        structured_stimuli, key=lambda usi_list: usi_list[0][1]['index_in_master_experiment_stimuli'])

    sentence_vectors = list()
    for usi_list in structured_stimuli:
        index_first_noun = [i for q, i in zip(load_data.is_first_noun(usi_list), range(len(usi_list))) if q]
        index_verb = [i for q, i in zip(load_data.is_first_non_to_be_verb(usi_list), range(len(usi_list))) if q]
        index_second_noun = [i for q, i in zip(load_data.is_second_noun(usi_list), range(len(usi_list))) if q]
        first_noun = \
            usi_list[index_first_noun[0]][1]['stimulus'] if len(index_first_noun) > 0 else None
        verb = \
            usi_list[index_verb[0]][1]['stimulus'] if len(index_verb) > 0 else None
        second_noun = \
            usi_list[index_second_noun[0]][1]['stimulus'] if len(index_second_noun) > 0 else None
        keys = [
            load_data.punctuation_regex.sub('', key).lower() if key is not None else None
            for key in [first_noun, verb, second_noun]]
        sentence_vectors.append([model_embeddings[key] if key is not None else None for key in keys])

    filled_vectors = list()
    for current_vectors in sentence_vectors:
        filled_vectors.append(np.mean([v for v in current_vectors if v is not None], axis=0))
    vectors = np.array(filled_vectors)


    model_rdm = squareform(pdist(vectors, metric=distance))
    return model_rdm



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
    len_labels = [LENGTH[voice_lab][word][pos_labels[i_lab]] for i_lab, voice_lab in enumerate(voice_labels)]
    syn_rdm = make_syntax_rdm(len_labels, voice_labels)

    bow_rdm = load_bow(experiment, dist)
    hier_rdm = np.load(MODEL_PATH.format(experiment=experiment, model='hierarchical')).item()['rdm']

    return np.concatenate(subject_val_rdms, axis=0), np.concatenate(subject_test_rdms, axis=0), \
           np.concatenate(subject_total_rdms, axis=0), syn_rdm, bow_rdm, hier_rdm, time


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
    parser.add_argument('--win_len', type=int, default=2)
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

    sub_val_rdms, sub_test_rdms, sub_total_rdms, syn_rdm, bow_rdm, hier_rdm, time = load_all_rdms(experiment,
                                                                                                  word,
                                                                                                  win_len,
                                                                                                  overlap,
                                                                                                  dist,
                                                                                                  doTimeAvg)



    syn_bow_scores, _ = ktau_rdms(bow_rdm, syn_rdm)
    print('Correlation between BoW and Syntax RDMs is: {}'.format(syn_bow_scores))

    syn_hier_scores, _ = ktau_rdms(hier_rdm, syn_rdm)
    print('Correlation between Hierarchical and Syntax RDMs is: {}'.format(syn_hier_scores))

    hier_bow_scores, _ = ktau_rdms(bow_rdm, hier_rdm)
    print('Correlation between BoW and Hierarchical RDMs is: {}'.format(hier_bow_scores))


    rdm_fig = plt.figure(figsize=(16, 8))
    rdm_grid = AxesGrid(rdm_fig, 111, nrows_ncols=(1, 3),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True, aspect=True)

    rdms_to_plot = [syn_rdm, bow_rdm/np.max(bow_rdm), hier_rdm/np.max(hier_rdm)]
    titles_to_plot = ['Syntax', 'Bag of Words', 'Hierarchical']

    for i_rdm, rdm in enumerate(rdms_to_plot):
        im = rdm_grid[i_rdm].imshow(rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
        rdm_grid[i_rdm].set_title(titles_to_plot[i_rdm], fontsize=axistitlesize)
        rdm_grid[i_rdm].text(-0.15, 1.05, string.ascii_uppercase[i_rdm], transform=rdm_grid[i_rdm].transAxes,
                size=axislettersize, weight='bold')
    cbar = rdm_grid.cbar_axes[0].colorbar(im)
    rdm_fig.suptitle('RDM Comparison')
    rdm_fig.savefig(SAVE_FIG.format(fig_type='rdm-comp-models',
                                      word=word,
                                      win_len=win_len,
                                      ov=overlap,
                                      dist=dist,
                                      avgTm=doTimeAvg) + '.png', bbox_inches='tight')
    rdm_fig.savefig(SAVE_FIG.format(fig_type='rdm-comp-models',
                                    word=word,
                                    win_len=win_len,
                                    ov=overlap,
                                    dist=dist,
                                    avgTm=doTimeAvg) + '.pdf', bbox_inches='tight')

    val_rdms = np.squeeze(np.mean(sub_val_rdms, axis=0))
    test_rdms = np.squeeze(np.mean(sub_test_rdms, axis=0))
    total_avg_rdms = np.squeeze(np.mean(sub_total_rdms, axis=0))
    num_sub = sub_total_rdms.shape[0]
    print(num_sub)
    num_time = test_rdms.shape[1]

    noise_rep_lb_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='noise-rep-lb',
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=doTimeAvg)
    if os.path.isfile(noise_rep_lb_file) and not force:
        result = np.load(noise_rep_lb_file)
        noise_rep_lb_ceiling = result['scores']
    else:
        noise_rep_lb_ceiling = score_rdms(val_rdms, test_rdms)
        np.savez_compressed(noise_rep_lb_file, scores=noise_rep_lb_ceiling)

    noise_rep_ub_file = SAVE_SCORES.format(exp=experiment,
                                           score_type='noise-rep-ub',
                                           word=word,
                                           win_len=win_len,
                                           ov=overlap,
                                           dist=dist,
                                           avgTm=doTimeAvg)
    if os.path.isfile(noise_rep_ub_file) and not force:
        result = np.load(noise_rep_ub_file)
        noise_rep_ub_ceiling = result['scores']
    else:
        noise_rep_ub_ceiling = score_rdms(val_rdms, total_avg_rdms)
        np.savez_compressed(noise_rep_ub_file, scores=noise_rep_ub_ceiling)

    mean_noise_rep_lb = np.squeeze(np.mean(noise_rep_lb_ceiling, axis=0))
    std_noise_rep_lb = np.squeeze(np.std(noise_rep_lb_ceiling, axis=0))

    noise_lb = np.max(mean_noise_rep_lb - std_noise_rep_lb)

    mean_noise_rep_ub = np.squeeze(np.mean(noise_rep_ub_ceiling, axis=0))
    std_noise_rep_ub = np.squeeze(np.std(noise_rep_ub_ceiling, axis=0))

    noise_ub = np.max(mean_noise_rep_ub + std_noise_rep_ub)

    bow_rep_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='bow-rep',
                                        word=word,
                                        win_len=win_len,
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvg)
    if os.path.isfile(bow_rep_file) and not force:
        result = np.load(bow_rep_file)
        bow_rep_scores = result['scores']
    else:
        bow_rep_scores = score_rdms(bow_rdm, total_avg_rdms)
        np.savez_compressed(bow_rep_file, scores=bow_rep_scores)

    hier_rep_file = SAVE_SCORES.format(exp=experiment,
                                      score_type='hier-rep',
                                      word=word,
                                      win_len=win_len,
                                      ov=overlap,
                                      dist=dist,
                                      avgTm=doTimeAvg)
    if os.path.isfile(hier_rep_file) and not force:
        result = np.load(hier_rep_file)
        hier_rep_scores = result['scores']
    else:
        hier_rep_scores = score_rdms(hier_rdm, total_avg_rdms)
        np.savez_compressed(hier_rep_file, scores=hier_rep_scores)

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

    plot_time = time + args.win_len * 0.002

    rep_fig, rep_ax = plt.subplots()

    rep_ax.plot(plot_time, syn_rep_scores, label='Syntax', color='r')
    rep_ax.plot(plot_time, bow_rep_scores, label='Bag of Words', color='b')
    rep_ax.plot(plot_time, hier_rep_scores, label='Hierarchical', color='g')

    rep_ax.fill_between(plot_time, mean_noise_rep_lb - std_noise_rep_lb, mean_noise_rep_ub + std_noise_rep_ub,
                        facecolor='0.5', alpha=0.5, edgecolor='w')
    rep_ax.legend(loc=1, fontsize=legendfontsize)

    rep_ax.set_ylim([0.0, 0.8])
    rep_ax.set_xlim([np.min(plot_time), np.max(plot_time)])

    #
    # xticklabels_to_plot = ['Full', 'Partial']
    # scores_to_plot = [[np.max(syn_rep_scores), np.max(syn_rep_cond_scores)],
    #                   [np.max(pos_rep_scores), np.max(pos_rep_cond_scores)]]
    # cond_fig, ax = plt.subplots(figsize=(12, 8))
    #
    # ind = np.arange(len(xticklabels_to_plot))
    # width = 0.25  # the width of the bars
    #
    # ax.fill_between([ind[0], ind[-1] + 2.0*width], [noise_lb, noise_lb],
    #                 [noise_ub, noise_ub],
    #                 facecolor='0.5', alpha=0.5, edgecolor='w')
    # ax.bar(ind, scores_to_plot[0], width,
    #        color='r', label='Syntax')
    # ax.bar(ind + width, scores_to_plot[1], width,
    #        color='b', label='Length')
    # ax.legend(fontsize=legendfontsize)
    # ax.set_ylabel('Kendall tau', fontsize=axislabelsize)
    # ax.set_ylim([0.0, 0.8])
    # ax.set_xticks(ind + width)
    # ax.set_xticklabels(xticklabels_to_plot)
    # ax.set_xlim([0.0, ind[-1] + 2.0*width])
    # ax.set_xlabel('Correlation Type', fontsize=axislabelsize)
    # ax.tick_params(labelsize=ticklabelsize)
    #
    # cond_fig.suptitle('Correlation Type Comparison', fontsize=suptitlesize)
    # cond_fig.subplots_adjust(top=0.85)
    # cond_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-cond-comp-pos',
    #                                   word=word,
    #                                   win_len=win_len,
    #                                   ov=overlap,
    #                                   dist=dist,
    #                                   avgTm=doTimeAvg) + '.png', bbox_inches='tight')
    #
    # cond_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-cond-comp-pos',
    #                                  word=word,
    #                                  win_len=win_len,
    #                                  ov=overlap,
    #                                  dist=dist,
    #                                  avgTm=doTimeAvg) + '.pdf', bbox_inches='tight')

    plt.show()