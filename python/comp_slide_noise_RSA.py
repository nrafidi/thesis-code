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

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_{fig_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}'
SAVE_SCORES = '/share/volume0/nrafidi/RSA_scores/{exp}/RSA_{score_type}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}.npz'

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


TEXT_PAD_X = -0.125
TEXT_PAD_Y = 1.02

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

    pos_labels = [POS[lab] for lab in labels]
    pos_rdm = make_model_rdm(pos_labels)
    len_labels = [LENGTH[voice_lab][word][pos_labels[i_lab]] for i_lab, voice_lab in enumerate(voice_labels)]
    syn_rdm = make_syntax_rdm(len_labels, voice_labels)
    voice_rdm = make_model_rdm(voice_labels)
    word_rdm = make_model_rdm(labels)

    return np.concatenate(subject_val_rdms, axis=0), np.concatenate(subject_test_rdms, axis=0), \
           np.concatenate(subject_total_rdms, axis=0), word_rdm, voice_rdm, pos_rdm, syn_rdm, time


# assuming draw x time x stim x stim
def score_rdms(val_rdms, test_rdms):
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
            scores[i_draw, i_time], _ = ktau_rdms(val, test)
    return np.squeeze(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--overlap', type=int, default=2)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    experiment = args.experiment
    overlap = args.overlap
    dist = args.dist

    force = args.force

    win_lens = [2, 25, 50, 100]
    doTimeAvgs = ['T', 'F']
    word = 'eos-full'

    sub_val_rdms, sub_test_rdms, sub_total_rdms, word_rdm, voice_rdm, pos_rdm, syn_rdm, time = load_all_rdms(experiment,
                                                                                                              word,
                                                                                                              win_lens[0],
                                                                                                              overlap,
                                                                                                              dist,
                                                                                                              doTimeAvgs[1])


    val_rdms = np.squeeze(np.mean(sub_val_rdms, axis=0))
    test_rdms = np.squeeze(np.mean(sub_test_rdms, axis=0))
    total_avg_rdms = np.squeeze(np.mean(sub_total_rdms, axis=0))
    num_sub = sub_total_rdms.shape[0]
    print(num_sub)
    num_time = test_rdms.shape[1]

    noise_rep_lb_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='noise-rep-lb',
                                            word=word,
                                            win_len=win_lens[0],
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=doTimeAvgs[1])
    if os.path.isfile(noise_rep_lb_file) and not force:
        result = np.load(noise_rep_lb_file)
        noise_rep_lb_ceiling = result['scores']
    else:
        noise_rep_lb_ceiling = score_rdms(val_rdms, test_rdms)
        np.savez_compressed(noise_rep_lb_file, scores=noise_rep_lb_ceiling)

    noise_sub_lb_file = SAVE_SCORES.format(exp=experiment,
                                           score_type='noise-sub-lb',
                                           word=word,
                                           win_len=win_lens[0],
                                           ov=overlap,
                                           dist=dist,
                                           avgTm=doTimeAvgs[1])
    if os.path.isfile(noise_sub_lb_file) and not force:
        result = np.load(noise_sub_lb_file)
        noise_sub_lb_ceiling = result['scores']
    else:
        noise_sub_lb_ceiling = []
        for i_sub in range(num_sub):
            sub_inds = np.logical_not(np.arange(num_sub) == i_sub)
            sub_score = score_rdms(np.squeeze(sub_total_rdms[i_sub, ...]), np.squeeze(np.mean(sub_total_rdms[sub_inds, ...], axis=0)))
            noise_sub_lb_ceiling.append(sub_score[None, ...])
        noise_sub_lb_ceiling = np.concatenate(noise_sub_lb_ceiling, axis=0)
        np.savez_compressed(noise_sub_lb_file, scores=noise_sub_lb_ceiling)

    noise_rep_ub_file = SAVE_SCORES.format(exp=experiment,
                                           score_type='noise-rep-ub',
                                           word=word,
                                           win_len=win_lens[0],
                                           ov=overlap,
                                           dist=dist,
                                           avgTm=doTimeAvgs[1])
    if os.path.isfile(noise_rep_ub_file) and not force:
        result = np.load(noise_rep_ub_file)
        noise_rep_ub_ceiling = result['scores']
    else:
        noise_rep_ub_ceiling = score_rdms(val_rdms, total_avg_rdms)
        np.savez_compressed(noise_rep_ub_file, scores=noise_rep_ub_ceiling)

    noise_sub_ub_file = SAVE_SCORES.format(exp=experiment,
                                           score_type='noise-sub-ub',
                                           word=word,
                                           win_len=win_lens[0],
                                           ov=overlap,
                                           dist=dist,
                                           avgTm=doTimeAvgs[1])
    if os.path.isfile(noise_sub_ub_file) and not force:
        result = np.load(noise_sub_ub_file)
        noise_sub_ub_ceiling = result['scores']
    else:
        noise_sub_ub_ceiling = []
        for i_sub in range(num_sub):
            sub_inds = np.logical_not(np.arange(num_sub) == i_sub)
            sub_score = score_rdms(np.squeeze(sub_total_rdms[i_sub, ...]), total_avg_rdms)
            noise_sub_ub_ceiling.append(sub_score[None, ...])
        noise_sub_ub_ceiling = np.concatenate(noise_sub_ub_ceiling, axis=0)
        np.savez_compressed(noise_sub_ub_file, scores=noise_sub_ub_ceiling)


    mean_noise_rep_lb = np.squeeze(np.mean(noise_rep_lb_ceiling, axis=0))
    std_noise_rep_lb = np.squeeze(np.std(noise_rep_lb_ceiling, axis=0))

    mean_noise_rep_ub = np.squeeze(np.mean(noise_rep_ub_ceiling, axis=0))
    std_noise_rep_ub = np.squeeze(np.std(noise_rep_ub_ceiling, axis=0))

    mean_noise_sub_lb = np.squeeze(np.mean(noise_sub_lb_ceiling, axis=0))
    std_noise_sub_lb = np.squeeze(np.std(noise_sub_lb_ceiling, axis=0))

    mean_noise_sub_ub = np.squeeze(np.mean(noise_sub_ub_ceiling, axis=0))
    std_noise_sub_ub = np.squeeze(np.std(noise_sub_ub_ceiling, axis=0))

    plot_time = time + win_lens[0]*0.002

    pos_rep_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='pos-rep',
                                        word=word,
                                        win_len=win_lens[0],
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvgs[1])
    if os.path.isfile(pos_rep_file) and not force:
        result = np.load(pos_rep_file)
        pos_rep_scores = result['scores']
    else:
        pos_rep_scores = score_rdms(pos_rdm, total_avg_rdms)
        np.savez_compressed(pos_rep_file, scores=pos_rep_scores)

    pos_sub_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='pos-sub',
                                        word=word,
                                        win_len=win_lens[0],
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvgs[1])
    if os.path.isfile(pos_sub_file) and not force:
        result = np.load(pos_sub_file)
        pos_sub_scores = result['scores']
    else:
        pos_sub_scores = []
        for i_sub in range(num_sub):
            sub_score = score_rdms(pos_rdm, np.squeeze(sub_total_rdms[i_sub, ...]))
            pos_sub_scores.append(sub_score[None, ...])
        pos_sub_scores = np.concatenate(pos_sub_scores, axis=0)
        np.savez_compressed(pos_sub_file, scores=pos_sub_scores)
    mean_pos = np.squeeze(np.mean(pos_sub_scores, axis=0))
    std_pos = np.squeeze(np.std(pos_sub_scores, axis=0))

    syn_rep_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='syn-rep',
                                        word=word,
                                        win_len=win_lens[0],
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvgs[1])
    if os.path.isfile(syn_rep_file) and not force:
        result = np.load(syn_rep_file)
        syn_rep_scores = result['scores']
    else:
        syn_rep_scores = score_rdms(syn_rdm, total_avg_rdms)
        np.savez_compressed(syn_rep_file, scores=syn_rep_scores)

    syn_sub_file = SAVE_SCORES.format(exp=experiment,
                                        score_type='syn-sub',
                                        word=word,
                                        win_len=win_lens[0],
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvgs[1])
    if os.path.isfile(syn_sub_file) and not force:
        result = np.load(syn_sub_file)
        syn_sub_scores = result['scores']
    else:
        syn_sub_scores = []
        for i_sub in range(num_sub):
            sub_score = score_rdms(syn_rdm, np.squeeze(sub_total_rdms[i_sub, ...]))
            syn_sub_scores.append(sub_score[None, ...])
        syn_sub_scores = np.concatenate(syn_sub_scores, axis=0)
        np.savez_compressed(syn_sub_file, scores=syn_sub_scores)
    mean_syn = np.squeeze(np.mean(syn_sub_scores, axis=0))
    std_syn = np.squeeze(np.std(syn_sub_scores, axis=0))

    noise_fig = plt.figure(figsize=(12, 8))
    noise_grid = AxesGrid(noise_fig, 111, nrows_ncols=(1, 2),
                          axes_pad=0.4)

    sub_ax = noise_grid[0]
    rep_ax = noise_grid[1]

    sub_ax.plot(plot_time, mean_syn, label='Syntax', color='r')
    sub_ax.fill_between(plot_time, mean_syn - std_syn, mean_syn + std_syn,
                    facecolor='r', alpha=0.5, edgecolor='w')
    sub_ax.plot(plot_time, mean_pos, label='POS', color='b')
    sub_ax.fill_between(plot_time, mean_pos - std_pos, mean_pos + std_pos,
                    facecolor='b', alpha=0.5, edgecolor='w')

    sub_ax.fill_between(plot_time, mean_noise_sub_lb - std_noise_sub_lb, mean_noise_sub_ub + std_noise_sub_ub,
                    facecolor='0.5', alpha=0.5, edgecolor='w')
    sub_ax.legend(loc=1)
    sub_ax.set_ylabel('Kendall Tau', fontsize=18)
    
    sub_ax.set_title('Subject Noise Ceiling')
    sub_ax.text(TEXT_PAD_X, TEXT_PAD_Y, string.ascii_uppercase[0], transform=sub_ax.transAxes,
            size=20, weight='bold')
    sub_ax.set_ylim([0.0, 0.6])
    sub_ax.set_xlim([np.min(plot_time), np.max(plot_time)])

    rep_ax.plot(plot_time, syn_rep_scores, label='Syntax', color='r')
    rep_ax.plot(plot_time, pos_rep_scores, label='POS', color='b')

    rep_ax.fill_between(plot_time, mean_noise_rep_lb - std_noise_rep_lb, mean_noise_rep_ub + std_noise_rep_ub,
                        facecolor='0.5', alpha=0.5, edgecolor='w')
    rep_ax.legend(loc=1)

    rep_ax.set_title('Repetition Noise Ceiling')
    rep_ax.text(TEXT_PAD_X, TEXT_PAD_Y, string.ascii_uppercase[1], transform=rep_ax.transAxes,
                size=20, weight='bold')
    rep_ax.set_ylim([0.0, 0.6])
    rep_ax.set_xlim([np.min(plot_time), np.max(plot_time)])

    noise_fig.suptitle('Noise Ceiling Comparison', fontsize=25)

    noise_fig.text(0.5, 0.04, 'Time Relative to Last Word Onset (s)', ha='center', fontsize=18)
    noise_fig.subplots_adjust(top=0.85)
    noise_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-noise-comp',
                                        word=word,
                                        win_len=win_lens[0],
                                        ov=overlap,
                                        dist=dist,
                                        avgTm=doTimeAvgs[1]) + '.png', bbox_inches='tight')
    noise_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-noise-comp',
                                      word=word,
                                      win_len=win_lens[0],
                                      ov=overlap,
                                      dist=dist,
                                      avgTm=doTimeAvgs[1]) + '.pdf', bbox_inches='tight')

    avg_time_strs = ['Time Average', 'No Time Average']
    win_fig, ax = plt.subplots(figsize=(20, 20))
    colors = ['0.2', '0.8']
    ind = np.arange(len(win_lens))  # the x locations for the groups
    width = 0.25  # the width of the bars
    for i_avg, avg in enumerate(doTimeAvgs):
        win_len_comp_noise_lb = np.empty((len(win_lens)))
        win_len_comp_noise_ub = np.empty((len(win_lens)))
        win_labels = []
        for i_win, win in enumerate(win_lens):

            sub_val_rdms, sub_test_rdms, sub_total_rdms, word_rdm, voice_rdm, pos_rdm, syn_rdm, time = load_all_rdms(
                experiment,
                word,
                win,
                overlap,
                dist,
                avg)

            val_rdms = np.squeeze(np.mean(sub_val_rdms, axis=0))
            test_rdms = np.squeeze(np.mean(sub_test_rdms, axis=0))
            total_avg_rdms = np.squeeze(np.mean(sub_total_rdms, axis=0))

            noise_rep_lb_file = SAVE_SCORES.format(exp=experiment,
                                                   score_type='noise-rep-lb',
                                                   word=word,
                                                   win_len=win,
                                                   ov=overlap,
                                                   dist=dist,
                                                   avgTm=avg)
            if os.path.isfile(noise_rep_lb_file) and not force:
                result = np.load(noise_rep_lb_file)
                noise_rep_lb_ceiling = result['scores']
            else:
                noise_rep_lb_ceiling = score_rdms(val_rdms, test_rdms)
                np.savez_compressed(noise_rep_lb_file, scores=noise_rep_lb_ceiling)
            mean_noise_rep_lb = np.squeeze(np.mean(noise_rep_lb_ceiling, axis=0))
            std_noise_rep_lb = np.squeeze(np.std(noise_rep_lb_ceiling, axis=0))
            win_len_comp_noise_lb[i_win] = np.max(mean_noise_rep_lb - std_noise_rep_lb)

            noise_rep_ub_file = SAVE_SCORES.format(exp=experiment,
                                                   score_type='noise-rep-ub',
                                                   word=word,
                                                   win_len=win,
                                                   ov=overlap,
                                                   dist=dist,
                                                   avgTm=avg)
            if os.path.isfile(noise_rep_ub_file) and not force:
                result = np.load(noise_rep_ub_file)
                noise_rep_ub_ceiling = result['scores']
            else:
                noise_rep_ub_ceiling = score_rdms(val_rdms, total_avg_rdms)
                np.savez_compressed(noise_rep_ub_file, scores=noise_rep_ub_ceiling)
            mean_noise_rep_ub = np.squeeze(np.mean(noise_rep_ub_ceiling, axis=0))
            std_noise_rep_ub = np.squeeze(np.std(noise_rep_ub_ceiling, axis=0))
            win_len_comp_noise_ub[i_win] = np.max(mean_noise_rep_ub + std_noise_rep_ub)

            win_in_s = win * 0.002
            label_str = '%.3f s' % win_in_s
            win_labels.append(label_str)

        win_len_comp_noise = (win_len_comp_noise_lb + win_len_comp_noise_ub)/2.0
        ax.bar(ind + i_avg*width, win_len_comp_noise, width,
               color=colors[i_avg], label=avg_time_strs[i_avg], yerr=[win_len_comp_noise_lb, win_len_comp_noise_ub],
               ecolor=colors[1 - avg])
    ax.legend()
    ax.set_ylabel('Noise Ceiling Midpoint')
    ax.set_ylim([0.0, 1.25])
    ax.set_xlim([ind[0], ind[-1] + 2.0*width])
    ax.set_xticks(ind + width)
    ax.set_xticklabels(win_labels)

    win_fig.suptitle('Window Size Comparison', fontsize=25)

    win_fig.text(0.5, 0.04, 'Window Size (s)', ha='center', fontsize=18)
    win_fig.subplots_adjust(top=0.85)
    win_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-win-comp',
                                      word=word,
                                      win_len='all',
                                      ov=overlap,
                                      dist=dist,
                                      avgTm='TF') + '.png', bbox_inches='tight')
    win_fig.savefig(SAVE_FIG.format(fig_type='score-overlay-win-comp',
                                      word=word,
                                      win_len='all',
                                      ov=overlap,
                                      dist=dist,
                                      avgTm='TF') + '.pdf', bbox_inches='tight')

    plt.show()