import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import string
import tgm_kf_acc_eos_multisub
import tgm_loso_acc_eos_multisub

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice',
                   'propid': 'Proposition ID',
                   'senlen': 'Sentence Length',
                   'bind':'Argument Binding'}

PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='T')
    parser.add_argument('--avgTest', default='T')
    args = parser.parse_args()

    win_len = args.win_len
    num_instances = args.num_instances

    ticklabelsize = 14
    legendfontsize = 28
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20
    time_step = int(250 / args.overlap)

    colors = ['r', 'b', 'g', 'k']
    sen_type = 'pooled'
    experiment = 'krns2'
    word_list = ['verb', 'voice', 'propid']
    num_folds = [2, 4, 8]
    cv_random_states = range(100)
    chance = 0.5
    time_adjust = win_len * 0.002

    sen_fig, sen_axs = plt.subplots(1, len(word_list), figsize=(36, 12))
    for i_word, word in enumerate(word_list):

        ax = sen_axs[i_word]

        for i_k, k in enumerate(num_folds):
            if k > 2 and word == 'propid':
                break
            top_dir = tgm_kf_acc_eos_multisub.TOP_DIR.format(exp=experiment)


            fold_diags = []
            for rsCV in cv_random_states:

                rank_file = tgm_kf_acc_eos_multisub.MULTI_SAVE_FILE.format(dir=top_dir,
                                                                             sen_type=sen_type,
                                                                             word=word,
                                                                             win_len=args.win_len,
                                                                           k=k,
                                                                           rsCV=rsCV,
                                                                             ov=args.overlap,
                                                                             perm='F',
                                                                             alg=args.alg,
                                                                             adj=args.adj,
                                                                             avgTm=args.avgTime,
                                                                             avgTst=args.avgTest,
                                                                             inst=args.num_instances,
                                                                             rsP=1,
                                                                             rank_str='rank',
                                                                             mode='acc')

                rank_result = np.load(rank_file + '.npz')
                acc_all = rank_result['tgm_rank']
                diag_acc = np.diag(np.mean(acc_all, axis=0))
                fold_diags.append(diag_acc[None, ...])

            fold_diag = np.mean(np.concatenate(fold_diags, axis=0), axis=0)

            ax.plot(fold_diag, label=k, color=colors[i_k], linewidth=2.0)

        top_dir = tgm_loso_acc_eos_multisub.TOP_DIR.format(exp=experiment)
        multi_file = tgm_loso_acc_eos_multisub.MULTI_SAVE_FILE.format(dir=top_dir,
                                                                      sen_type=sen_type,
                                                                      word=word,
                                                                      win_len=args.win_len,
                                                                      ov=args.overlap,
                                                                      perm='F',
                                                                      exc='',
                                                                      alg=args.alg,
                                                                      adj=args.adj,
                                                                      avgTm=args.avgTime,
                                                                      avgTst=args.avgTest,
                                                                      inst=args.num_instances,
                                                                      rsP=1,
                                                                      rank_str='',
                                                                      mode='acc')

        rank_file = tgm_loso_acc_eos_multisub.MULTI_SAVE_FILE.format(dir=top_dir,
                                                                     sen_type=sen_type,
                                                                     word=word,
                                                                     win_len=args.win_len,
                                                                     ov=args.overlap,
                                                                     exc='',
                                                                     perm='F',
                                                                     alg=args.alg,
                                                                     adj=args.adj,
                                                                     avgTm=args.avgTime,
                                                                     avgTst=args.avgTest,
                                                                     inst=args.num_instances,
                                                                     rsP=1,
                                                                     rank_str='rank',
                                                                     mode='acc')

        rank_result = np.load(rank_file + '.npz')
        acc_all = rank_result['tgm_rank']
        result = np.load(multi_file + '.npz')
        time = result['time']
        win_starts = result['win_starts']
        diag_acc = np.diag(np.mean(acc_all, axis=0))

        ax.plot(diag_acc, label='LOO', color=colors[3], linewidth=2.0)

        num_time = len(win_starts)
        max_line = 0.3 * 2 * time_step

        ax.set_xticks(range(0, len(time[win_starts]), time_step))
        label_time = time[win_starts]
        label_time = label_time[::time_step]
        label_time[np.abs(label_time) < 1e-15] = 0.0

        ax.axhline(y=chance, color='k', linestyle='dashed')

        ax.set_xticklabels(label_time)
        ax.axvline(x=max_line, color='k')
        if i_word == 0:
            ax.set_ylabel('Rank Accuracy', fontsize=axislabelsize)
        if i_word == 1:
            ax.set_xlabel('Time Relative to Last Word Onset (s)', fontsize=axislabelsize)
        ax.set_ylim([0.0, 1.2])
        ax.set_xlim([0, len(time[win_starts])])
        ax.tick_params(labelsize=ticklabelsize)
        if i_word == 0:
            ax.legend(loc=3, ncol=2, fontsize=legendfontsize)
        ax.set_title('{word}'.format(word=PLOT_TITLE_WORD[word]), fontsize=axistitlesize)
        ax.text(-0.05, 1.05, string.ascii_uppercase[i_word], transform=ax.transAxes,
                size=axislettersize, weight='bold')

    sen_fig.subplots_adjust(top=0.85)
    sen_fig.suptitle('Rank Accuracy Over Time\nPost-Sentence', fontsize=suptitlesize)
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos-test-kf_diag_acc_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            exp=experiment, sen_type='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=win_len,
            overlap=args.overlap,
            num_instances=num_instances,
        ), bbox_inches='tight')
    sen_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos-test-kf_diag_acc_multisub_{sen_type}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=experiment, sen_type='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=win_len,
            overlap=args.overlap,
            num_instances=num_instances,
        ), bbox_inches='tight')

    plt.show()


