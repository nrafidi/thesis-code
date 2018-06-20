import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import string
import os
from rank_from_pred import rank_from_pred


TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
MULTI_SAVE_FILE = '{dir}TGM-LOSO{tsss}_multisub{short}{exc}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{rank_str}{mode}'

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                    'noun2': 'Second Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='T')
    parser.add_argument('--avgTest', default='T')
    parser.add_argument('--exc', action='store_true')
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--tsss', action='store_true')
    args = parser.parse_args()

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    if args.avgTime == 'T':
        avg_time_str = 'Time Average'
    else:
        avg_time_str = 'No Time Average'

    if args.avgTest == 'T':
        avg_test_str = 'Test Sample Average'
    else:
        avg_test_str = 'No Test Sample Average'

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    top_dir = TOP_DIR.format(exp=args.experiment)

    if args.exc:
        exc_str = '_exc'
    else:
        exc_str = ''

    if args.tsss:
        tsss_str = '_tsss'
    else:
        tsss_str = ''

    word_list = ['noun1', 'verb', 'noun2']
    num_plots = len(word_list)
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(6*num_plots, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, num_plots),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True)
    i_combo = 0
    for i_sen, sen_type in enumerate(['active', 'passive']):
        for i_word, word in enumerate(word_list):
            if args.short and word == 'noun2':
                short_str = '_short'
            else:
                short_str = ''


            multi_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                                sen_type=sen_type,
                                                word=word,
                                                win_len=args.win_len,
                                                short=short_str,
                                                exc=exc_str,
                                                tsss=tsss_str,
                                                ov=args.overlap,
                                                perm='F',
                                                alg=args.alg,
                                                adj=args.adj,
                                                avgTm=args.avgTime,
                                                avgTst=args.avgTest,
                                                inst=args.num_instances,
                                                rsP=1,
                                                rank_str='',
                                                mode='acc')
            rank_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                                sen_type=sen_type,
                                                word=word,
                                                win_len=args.win_len,
                                                exc=exc_str,
                                                tsss=tsss_str,
                                                short=short_str,
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

            result = np.load(multi_file + '.npz')
            if os.path.isfile(rank_file + '.npz'): # and word != 'noun2':
                rank_result = np.load(rank_file + '.npz')
                acc_all = rank_result['tgm_rank']
            else:
                tgm_pred = result['tgm_pred']
                l_ints = result['l_ints']
                cv_membership = result['cv_membership']
                fold_labels = []
                for i in range(len(cv_membership)):
                    fold_labels.append(np.mean(l_ints[cv_membership[i]]))

                acc_all = rank_from_pred(tgm_pred, fold_labels)
                np.savez_compressed(rank_file, tgm_rank=acc_all)
           # acc_all = result['tgm_acc']
            time = result['time']
            win_starts = result['win_starts']
            mean_acc = np.mean(acc_all, axis=0)
            meow = np.argmax(np.diag(mean_acc))
            print(meow)
            time_win = time[win_starts]

            if sen_type == 'active':
                text_to_write = np.array(['Det', 'Noun', 'Verb', 'Det', 'Noun.'])
                max_line = 2.51 * 2 * time_step - time_adjust
                start_line = - time_adjust
            else:
                text_to_write = np.array(['Det', 'Noun', 'was', 'Verb', 'by', 'Det', 'Noun.'])
                max_line = 3.51 * 2 * time_step - time_adjust
                start_line = - time_adjust

            if args.short:
                min_time = 1.0
                text_to_write = text_to_write[2:]
                max_line -= 3 * time_step
                if word != 'noun2':
                    mean_acc = mean_acc[4 * time_step:, :]
                    mean_acc = mean_acc[:, 4 * time_step:]
                    time_win = time_win[4 * time_step:]
            else:
                min_time = 0.0



            print(mean_acc.shape)
            print(np.max(mean_acc))
            num_time = len(time_win)

            ax = combo_grid[i_combo]
            print(i_combo)
            im = ax.imshow(mean_acc, interpolation='nearest', aspect='auto', vmin=0.5, vmax=1.00)

            ax.set_title('{word}'.format(
                word=PLOT_TITLE_WORD[word]), fontsize=axistitlesize)

            ax.set_xticks(np.arange(0.0, float(num_time), float(time_step)) - time_adjust)
            ax.set_yticks(np.arange(0, num_time, time_step) - time_adjust)
            if i_combo == len(word_list)*2 - 1:
                ax.set_xlim([0.0, float(num_time)])
                ax.set_ylim([float(num_time), 0.0])


            max_time = 0.5 * len(time_win) / time_step
            label_time = np.arange(min_time, max_time, 0.5)
            ax.set_xticklabels(label_time)

            ax.set_yticklabels(label_time)
            ax.tick_params(labelsize=ticklabelsize)
            for i_v, v in enumerate(np.arange(start_line, max_line + 5, time_step)):
                ax.axvline(x=v, color='w')
                if i_v == 0:
                    buff_space = 0.125
                else:
                    buff_space = 0.025
                if i_v < len(text_to_write):
                    ax.text(v + buff_space * 2*time_step, num_time - 0.1*time_step, text_to_write[i_v], color='w', fontsize=10)
            ax.text(-0.12, 1.02, string.ascii_uppercase[i_combo], transform=ax.transAxes,
                                    size=axislettersize, weight='bold')
            i_combo += 1
            #fig, meow_ax = plt.subplots()
            #meow_ax.plot(np.diag(mean_acc))
            #meow_ax.set_title('{}: {}'.format(sen_type, word))
        


    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('Rank Accuracy TGMs',
        fontsize=suptitlesize)
    combo_fig.text(0.04, 0.275, 'Train Time Relative to Sentence Onset (s)', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize)
    combo_fig.text(0.5, 0.04, 'Test Time Relative to Sentence Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.text(0.488, 0.8575, 'Active', ha='center', fontsize=axistitlesize+2)
    combo_fig.text(0.488, 0.475, 'Passive', ha='center', fontsize=axistitlesize+2)
    combo_fig.subplots_adjust(top=0.85)

    if args.short:
        short_str = '_short'
    else:
        short_str = ''

    combo_fig.savefig('/home/nrafidi/thesis_figs/{exp}_avg-tgm{tsss}_multisub{short}{exc}_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                    exp=args.experiment, sen_type='both', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    short=short_str,
                    tsss=tsss_str,
                    exc=exc_str,
                    num_instances=args.num_instances
                ), bbox_inches='tight')

    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_avg-tgm{tsss}_multisub{short}{exc}_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type='both', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            short=short_str,
            tsss=tsss_str,
            exc=exc_str,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    plt.show()

# Boneyard

# fig, ax = plt.subplots()
            # h = ax.imshow(np.squeeze(intersection), interpolation='nearest', aspect='auto', vmin=0, vmax=acc_all.shape[0])
            #
            # ax.set_ylabel('Train Time (s)')
            # ax.set_xlabel('Test Time (s)')
            # ax.set_title(
            #     'Intersection TGM\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
            #                                                               word=word,
            #                                                               experiment=args.experiment))
            # ax.set_xticks(range(0, len(time_win), time_step))
            # label_time = time_win
            # label_time = label_time[::time_step]
            # label_time[np.abs(label_time) < 1e-15] = 0.0
            # ax.set_xticklabels(label_time)
            # ax.set_yticks(range(0, len(time_win), time_step))
            # ax.set_yticklabels(label_time)
            # ax.set_xlim(left=time_step)
            # ax.set_ylim(top=time_step)
            #
            # for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
            #     ax.axvline(x=v, color='k')
            #     if i_v < len(text_to_write):
            #         plt.text(v + 0.05 * 2*time_step, time_step, text_to_write[i_v], color='k')
            # plt.colorbar(h)
            #
            # fig.tight_layout()
            # plt.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_intersection_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            #         win_len=args.win_len,
            #         overlap=args.overlap,
            #         num_instances=args.num_instances
            #     ), bbox_inches='tight')

# fig, ax = plt.subplots()
            # h = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=0.25, vmax=0.5)
            # ax.set_ylabel('Train Time (s)')
            # ax.set_xlabel('Test Time (s)')
            # ax.set_title('Average TGM Decoding {word} from {sen_type}\n{avgTime}, {avgTest}\nNumber of Instances: {ni}'.format(
            #     sen_type=PLOT_TITLE_SEN[sen_type],
            #     word=PLOT_TITLE_WORD[word],
            #     avgTime=avg_time_str,
            #     avgTest=avg_test_str,
            #     ni=args.num_instances))
            # ax.set_xticks(range(0, len(time_win), time_step))
            # label_time = time_win
            # label_time = label_time[::time_step]
            # label_time[np.abs(label_time) < 1e-15] = 0.0
            # ax.set_xticklabels(label_time)
            # ax.set_yticks(range(0, len(time_win), time_step))
            # ax.set_yticklabels(label_time)
            # time_adjust = args.win_len
            #
            # for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
            #     ax.axvline(x=v, color='w')
            #     if i_v < len(text_to_write):
            #         plt.text(v + 0.05 * 2 * time_step, 1.5 * time_step, text_to_write[i_v], color='w')
            # plt.colorbar(h)
            # ax.set_xlim(left=time_step)
            # ax.set_ylim(top=time_step)
            # fig.tight_layout()
            # plt.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_avg-tgm-title_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
            #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            #         win_len=args.win_len,
            #         overlap=args.overlap,
            #         num_instances=args.num_instances
            #     ), bbox_inches='tight')
            #
            #
            #
            #
            # time_adjust = args.win_len*0.002
            # fig, ax = plt.subplots()
            #
            # ax.plot(np.diag(mean_acc), label='Accuracy')
            # ax.plot(frac_sub, label='Fraction of Subjects > Chance')
            #
            #
            # ax.set_xticks(range(0, len(time_win), time_step))
            # label_time = time_win
            # label_time = label_time[::time_step]
            # label_time[np.abs(label_time) < 1e-15] = 0.0
            # ax.set_xticklabels(label_time)
            # for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
            #     ax.axvline(x=v, color='k')
            #     if i_v < len(text_to_write):
            #         plt.text(v + 0.05, 0.8, text_to_write[i_v])
            # ax.set_ylabel('Accuracy/Fraction > Chance')
            # ax.set_xlabel('Time')
            # ax.set_ylim([0.0, 1.0])
            # ax.set_xlim(left=time_step)
            # ax.legend(loc=4)
            # ax.set_title('Mean Acc over subjects and Frac > Chance\n{sen_type} {word} {experiment}'.format(sen_type=sen_type,
            #                                                                              word=word,
            #                                                                              experiment=args.experiment))
            #
            # fig.tight_layout()
            # plt.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_diag_acc_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            #         exp=args.experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            #         win_len=args.win_len,
            #         overlap=args.overlap,
            #         num_instances=args.num_instances
            #     ), bbox_inches='tight')


# if sen_type == 'active':
#     text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
#     max_line = 2.0 - time_adjust + 0.01
#     if word == 'noun1':
#         start_line = -0.5 - time_adjust
#     else:
#         start_line = -1.0 - time_adjust
#
# else:
#     text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
#     max_line = 3.0 - time_adjust + 0.01
#     if word == 'noun1':
#         start_line = -0.5 - time_adjust
#     else:
#         start_line = -1.5 - time_adjust
