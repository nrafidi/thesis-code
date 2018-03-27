import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import run_TGM_LOSO
import tgm_loso_acc

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_TGM_LOSO.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['noun1', 'verb', 'noun2'])
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='F')
    args = parser.parse_args()

    if args.sen_type == 'active':
        max_time = 2.0
    else:
        max_time = 3.0

    win_lens = [12, 25, 50, 100, 150]
    num_insts = [1, 2, 5, 10]

    frac_sub_eos = []
    frac_sub_tot = []
    mean_max_eos = []
    mean_max_tot = []
    mean_mean_eos = []
    mean_mean_tot = []
    per_sub_max_eos = []
    per_sub_max_tot = []
    per_sub_mean_tot = []
    per_sub_mean_eos = []
    for win_len in win_lens:
        time_adjust = win_len * 0.002

        frac_sub_eos_win = []
        frac_sub_tot_win = []
        mean_max_eos_win = []
        mean_max_tot_win = []
        mean_mean_eos_win = []
        mean_mean_tot_win = []
        per_sub_max_eos_win = []
        per_sub_max_tot_win = []
        per_sub_mean_tot_win = []
        per_sub_mean_eos_win = []
        for num_instances in num_insts:
            intersection, acc_all, time, win_starts, eos_max = tgm_loso_acc.intersect_accs(args.experiment,
                                                                                           args.sen_type,
                                                                                           args.word,
                                                                                           win_len=win_len,
                                                                                           overlap=12,
                                                                                           adj=args.adj,
                                                                                           num_instances=num_instances,
                                                                                           avgTime=args.avgTime,
                                                                                           avgTest=args.avgTest)
            time = np.squeeze(time)
            time_ind = np.where(time[win_starts] >= (max_time - time_adjust))
            time_ind = time_ind[0]

            frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
            mean_acc = np.diag(np.mean(acc_all, axis=0))

            mean_max_tot_win.append(np.max(mean_acc))
            mean_mean_tot_win.append(np.mean(mean_acc))
            frac_sub_tot_win.append(frac_sub[np.argmax(mean_acc)])

            mean_max_eos_win.append(np.max(mean_acc[time_ind]))
            mean_mean_eos_win.append(np.mean(mean_acc[time_ind]))
            frac_sub_eos_win.append(frac_sub[time_ind[np.argmax(mean_acc[time_ind])]])

            sub_eos_max = []
            sub_eos_mean = []
            sub_tot_max = []
            sub_tot_mean = []
            for i_sub in range(acc_all.shape[0]):
                diag_acc = np.diag(np.squeeze(acc_all[i_sub, :, :]))
                sub_tot_max.append(np.max(diag_acc))
                sub_tot_mean.append(np.mean(diag_acc))
                sub_eos_max.append(np.max(diag_acc[time_ind]))
                sub_eos_mean.append(np.mean(diag_acc[time_ind]))
            sub_tot_max = np.array(sub_tot_max)
            sub_tot_mean = np.array(sub_tot_mean)
            sub_eos_max = np.array(sub_eos_max)
            sub_eos_mean = np.array(sub_eos_mean)

            per_sub_max_eos_win.append(sub_eos_max[None, ...])
            per_sub_max_tot_win.append(sub_tot_max[None, ...])
            per_sub_mean_tot_win.append(sub_tot_mean[None, ...])
            per_sub_mean_eos_win.append(sub_eos_mean[None, ...])

        frac_sub_eos_win = np.array(frac_sub_eos_win)
        frac_sub_eos.append(frac_sub_eos_win[None, ...])

        frac_sub_tot_win = np.array(frac_sub_tot_win)
        frac_sub_tot.append(frac_sub_tot_win[None, ...])

        mean_max_eos_win = np.array(mean_max_eos_win)
        mean_max_eos.append(mean_max_eos_win[None, ...])

        mean_max_tot_win = np.array(mean_max_tot_win)
        mean_max_tot.append(mean_max_tot_win[None, ...])

        mean_mean_eos_win = np.array(mean_mean_eos_win)
        mean_mean_eos.append(mean_mean_eos_win[None, ...])

        mean_mean_tot_win = np.array(mean_mean_tot_win)
        mean_mean_tot.append(mean_mean_tot_win[None, ...])

        per_sub_max_eos_win = np.concatenate(per_sub_max_eos_win, axis=0)
        per_sub_max_eos.append(per_sub_max_eos_win[None, ...])

        per_sub_max_tot_win = np.concatenate(per_sub_max_tot_win, axis=0)
        per_sub_max_tot.append(per_sub_max_tot_win[None, ...])

        per_sub_mean_tot_win = np.concatenate(per_sub_mean_tot_win, axis=0)
        per_sub_mean_tot.append(per_sub_mean_tot_win[None, ...])

        per_sub_mean_eos_win = np.concatenate(per_sub_mean_eos_win, axis=0)
        per_sub_mean_eos.append(per_sub_mean_eos_win[None, ...])

    frac_sub_eos = np.concatenate(frac_sub_eos, axis=0)

    frac_sub_tot = np.concatenate(frac_sub_tot, axis=0)

    mean_max_eos = np.concatenate(mean_max_eos, axis=0)

    mean_max_tot = np.concatenate(mean_max_tot, axis=0)

    mean_mean_eos = np.concatenate(mean_mean_eos, axis=0)

    mean_mean_tot = np.concatenate(mean_mean_tot, axis=0)

    per_sub_max_eos = np.concatenate(per_sub_max_eos, axis=0)

    per_sub_max_tot = np.concatenate(per_sub_max_tot, axis=0)

    per_sub_mean_eos = np.concatenate(per_sub_mean_eos, axis=0)

    per_sub_mean_tot = np.concatenate(per_sub_mean_tot, axis=0)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    h00 = axs[0][0].imshow(frac_sub_tot, interpolation='nearest', vmin=0.5, vmax=1.0)
    axs[0][0].set_title('Fraction of Subjects > Chance\nGlobal Max Accuracy')
    fig.colorbar(h00, ax=axs[0][0])
    h01 = axs[0][1].imshow(frac_sub_eos, interpolation='nearest', vmin=0.5, vmax=1.0)
    axs[0][1].set_title('Fraction of Subjects > Chance\nPost-Sentence Max Accuracy')
    fig.colorbar(h01, ax=axs[0][1])
    h10 = axs[1][0].imshow(mean_max_tot, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[1][0].set_title('Global Max Accuracy')
    fig.colorbar(h10, ax=axs[1][0])
    h11 = axs[1][1].imshow(mean_max_eos, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[1][1].set_title('Post-Sentence Max Accuracy')
    fig.colorbar(h11, ax=axs[1][1])
    h20 = axs[2][0].imshow(mean_mean_tot, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[2][0].set_title('Global Mean Accuracy')
    fig.colorbar(h20, ax=axs[2][0])
    h21 = axs[2][1].imshow(mean_mean_eos, interpolation='nearest', vmin=0.25, vmax=1.0)
    axs[2][1].set_title('Post-Sentence Mean Accuracy')
    fig.colorbar(h21, ax=axs[2][1])

    for i in range(3):
        for j in range(2):
            axs[i][j].set_xticks(range(len(num_insts)))
            axs[i][j].set_xticklabels(num_insts)
            axs[i][j].set_yticks(range(len(win_lens)))
            axs[i][j].set_yticklabels(np.array(win_lens).astype('float') * 2)
            axs[i][j].set_xlabel('Number of Instances')
            axs[i][j].set_ylabel('Window Length (ms)')


    fig.suptitle('Accuracy and Consistency Scores\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                             args.word,
                                                                             args.avgTime,
                                                                             args.avgTest),
                 fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig, axs = plt.subplots(1, 2)
    h0 = axs[0].imshow(frac_sub_tot + mean_max_tot, interpolation='nearest', vmin=0.75, vmax=2.0)
    axs[0].set_title('Combined Global Score')
    fig.colorbar(h0, ax=axs[0], shrink=0.5)
    h1 = axs[1].imshow(frac_sub_eos + mean_max_eos, interpolation='nearest', vmin=0.75, vmax=2.0)
    axs[1].set_title('Combined Post-Sentence Score')
    fig.colorbar(h1, ax=axs[1], shrink=0.5)

    for i in range(2):
        axs[i].set_xticks(range(len(num_insts)))
        axs[i].set_xticklabels(num_insts)
        axs[i].set_yticks(range(len(win_lens)))
        axs[i].set_yticklabels(np.array(win_lens).astype('float') * 2)
        axs[i].set_xlabel('Number of Instances')
        axs[i].set_ylabel('Window Length (ms)')

    fig.suptitle('Combined Score\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                             args.word,
                                                                             args.avgTime,
                                                                             args.avgTest),
                 fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)


    num_sub = per_sub_max_tot.shape[2]
    half_sub = int(num_sub/2)
    fig, axs = plt.subplots(2, half_sub, figsize=(30, 10))
    for i_sub in range(num_sub):
        if i_sub < half_sub:
            i_ax = 0
        else:
            i_ax = 1
        j_ax = i_sub - half_sub*i_ax
        h = axs[i_ax][j_ax].imshow(np.squeeze(per_sub_max_tot[:, :, i_sub]), interpolation='nearest', vmin=0.25, vmax=1.0)
        fig.colorbar(h, ax=axs[i_ax][j_ax])
        axs[i_ax][j_ax].set_title(run_TGM_LOSO.VALID_SUBS[args.experiment][i_sub])
        axs[i_ax][j_ax].set_xticks(range(len(num_insts)))
        axs[i_ax][j_ax].set_xticklabels(num_insts)
        axs[i_ax][j_ax].set_yticks(range(len(win_lens)))
        axs[i_ax][j_ax].set_yticklabels(np.array(win_lens).astype('float') * 2)
        axs[i_ax][j_ax].set_xlabel('Number of Instances')
        axs[i_ax][j_ax].set_ylabel('Window Length (ms)')
    fig.suptitle('Per Subject Global Max\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                      args.word,
                                                                      args.avgTime,
                                                                      args.avgTest),
                 fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)

    num_sub = per_sub_max_eos.shape[2]
    half_sub = int(num_sub / 2)
    fig, axs = plt.subplots(2, half_sub, figsize=(30, 10))
    for i_sub in range(num_sub):
        if i_sub < half_sub:
            i_ax = 0
        else:
            i_ax = 1
        j_ax = i_sub - half_sub * i_ax
        h = axs[i_ax][j_ax].imshow(np.squeeze(per_sub_max_eos[:, :, i_sub]), interpolation='nearest', vmin=0.25,
                                   vmax=1.0)
        axs[i_ax][j_ax].set_title(run_TGM_LOSO.VALID_SUBS[args.experiment][i_sub])
        axs[i_ax][j_ax].set_xticks(range(len(num_insts)))
        axs[i_ax][j_ax].set_xticklabels(num_insts)
        axs[i_ax][j_ax].set_yticks(range(len(win_lens)))
        axs[i_ax][j_ax].set_yticklabels(np.array(win_lens).astype('float') * 2)
    fig.colorbar(h)
    axs[1][0].set_xlabel('Number of Instances')
    axs[1][0].set_ylabel('Window Length (ms)')
    fig.suptitle('Per Subject Post-Sentence Max\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                              args.word,
                                                                              args.avgTime,
                                                                              args.avgTest),
                 fontsize=24)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)



    plt.show()
    #
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_win_inst_comp_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest
    #     ), bbox_inches='tight')
    #
    #
    # fig, ax = plt.subplots()
    # h0 = ax.imshow(frac_sub_tot + mean_acc_tot, interpolation='nearest', vmin=0.75, vmax=2.0)
    # ax.set_xticks(range(len(num_insts)))
    # ax.set_xticklabels(num_insts)
    # ax.set_yticks(range(len(win_lens)))
    # ax.set_yticklabels(np.array(win_lens).astype('float') * 2)
    # ax.set_title('Frac Subjects > Chance + Max EOS Accuracy')
    # ax.set_xlabel('Number of Instances')
    # ax.set_ylabel('Window Length (ms)')
    # fig.colorbar(h0, ax=ax)
    # fig.suptitle('Post Sentence Maximum\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
    #                                                                          args.word,
    #                                                                          args.avgTime,
    #                                                                          args.avgTest))
    # plt.subplots_adjust(top=0.8)
    #
    # plt.savefig(
    #     '/home/nrafidi/thesis_figs/{exp}_win_inst_comp-sum_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
    #         exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest
    #     ), bbox_inches='tight')
    # print(arg_max_eos)
    # print(arg_max_tot)
    # print(np.squeeze(per_sub_max_eos[:, 0, :]))
    #
    #
    #
    # plt.show()


