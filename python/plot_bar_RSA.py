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

SAVE_FIG = '/home/nrafidi/thesis_figs/RSA_bar_{model}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_{full_str}_split.pdf'
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
              'third-last': 'Third-Last Word',
              'third-last-full': 'Third-Last Word',
              'second-last-full': 'Second-Last Word',
              'last-full': 'Last Word',
              'eos': 'Post Sentence',
              'eos-full': 'Post Sentence All'}

COND_TITLE = {'len': 'Word Length',
              'pos': 'Part-of-Speech',
              'word': 'Word ID',
              'syn': 'Syntax',
              'None': 'Nothing'}

MODEL_TITLES = ['Part of Speech', 'Syntax', 'Word ID', 'Word Length']

TEXT_PAD_X = -0.125
TEXT_PAD_Y = 1.02


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--win_len', type=int)
    parser.add_argument('--overlap', type=int, default=3)
    parser.add_argument('--dist', default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--doTimeAvg', default='True', choices=['True', 'False'])
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    experiment = args.experiment
    win_len = args.win_len
    overlap = args.overlap
    dist = args.dist
    doTimeAvg = run_slide_noise_RSA.str_to_bool(args.doTimeAvg)
    force = args.force

    cond_list = ['None', 'pos', 'syn', 'word', 'len']
    word_list = ['third-last-full', 'second-last-full', 'last-full', 'eos-full']
    full_str = 'all'
    colors = ['r', 'g', 'b', 'm', 'c', 'y']

    mean_scores = np.empty((len(word_list), len(cond_list), len(cond_list) + 1))
    for i_word, word in enumerate(word_list):
        for j_cond, cond in enumerate(cond_list):
            noise_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='noise',
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )
            result = np.load(noise_file)
            mean_scores[i_word, j_cond, 0] = np.mean(result['noise_ceiling'])


            noise_ub_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='noise-ub',
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )

            result = np.load(noise_ub_file)
            mean_scores[i_word, j_cond, 1] = np.mean(result['noise_ub_ceiling'])


            pos_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='pos-ub-cond-{}'.format(cond),
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )

            result = np.load(pos_file)
            mean_scores[i_word, j_cond, 2] = np.mean(result['pos_scores'])


            syn_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='syn-ub-cond-{}'.format(cond),
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )

            result = np.load(syn_file)
            mean_scores[i_word, j_cond, 3] = np.mean(result['syn_scores'])

        
        
        
            word_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='word-ub-cond-{}'.format(cond),
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )

            result = np.load(word_file)
            mean_scores[i_word, j_cond, 4] = np.mean(result['word_scores'])


            string_file = SAVE_SCORES.format(exp=experiment,
                                            score_type='string-len-ub-cond-{}'.format(cond),
                                            word=word,
                                            win_len=win_len,
                                            ov=overlap,
                                            dist=dist,
                                            avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg),
                                            full_str=full_str
                                            )

            result = np.load(string_file)
            mean_scores[i_word, j_cond, 5] = np.mean(result['string_scores'])

    ind = np.arange(len(word_list))  # the x locations for the groups
    width = 0.8/float(len(cond_list))  # the width of the bars


    for i_model in range(len(cond_list) - 1):
        model_scores = np.squeeze(mean_scores[:, :, i_model + 2])
        fig, ax = plt.subplots()
        for j_cond in range(len(cond_list)):
            if j_cond != i_model + 1:
                ax.bar(ind + j_cond*width, model_scores[:, j_cond], width,
                       color=colors[j_cond], label=COND_TITLE[cond_list[j_cond]])

        # add some text for labels, title and axes ticks
        for i in ind:
            ax.fill_between([i, i + 0.8], [mean_scores[i, 0, 0], mean_scores[i, 0, 0]],
                            [mean_scores[i, 0, 1], mean_scores[i, 0, 1]],
                            facecolor='0.5', alpha=0.5, edgecolor='w')
        ax.set_ylabel('Mean Correlation with Neural Data')
        ax.set_ylim([-0.1, 1.0])
        ax.set_title(MODEL_TITLES[i_model])
        ax.set_xticks(ind + float(len(cond_list)*width) / 2.0)
        ax.set_xticklabels([PLOT_TITLE[word] for word in word_list])
        ax.legend()
        fig.savefig(SAVE_FIG.format(model=cond_list[i_model + 1],
                                      win_len=win_len,
                                      ov=overlap,
                                      dist=dist,
                                      full_str=full_str,
                                      avgTm=run_slide_noise_RSA.bool_to_str(doTimeAvg)), bbox_inches='tight')

    plt.show()