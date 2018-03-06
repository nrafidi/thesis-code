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

    win_lens = [12, 25, 50] #, 100, 150]
    num_insts = [1, 2, 5, 10]

    frac_sub_tot = []
    mean_acc_tot = []
    arg_max_tot = []
    arg_max_eos = []
    per_sub_max_eos = []
    for win_len in win_lens:
        print(win_len)
        time_adjust = win_len * 0.002
        frac_sub_win = []
        mean_acc_win = []
        arg_max_tot_win = []
        arg_max_eos_win = []
        per_sub_max_eos_win = []
        for num_instances in num_insts:
            print(num_instances)
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
            # print(time_ind)
            frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
            mean_acc = np.diag(np.mean(acc_all, axis=0))

            max_mean_acc = np.max(mean_acc[time_ind])
            argmax_mean_acc = np.argmax(mean_acc[time_ind])
            # print(argmax_mean_acc)
            arg_max_eos_win.append(time_ind[argmax_mean_acc])

            arg_max_tot_win.append(np.argmax(mean_acc))

            max_frac_sub = frac_sub[time_ind]
            max_frac_sub = max_frac_sub[argmax_mean_acc]

            frac_sub_win.append(max_frac_sub)
            mean_acc_win.append(max_mean_acc)

            sub_eos_max = []
            for i_sub in range(acc_all.shape[0]):
                diag_acc = np.diag(np.squeeze(acc_all[i_sub, :, :]))
                argo = np.argmax(diag_acc[time_ind])
                sub_eos_max.append(time_ind[argo])
            sub_eos_max = np.array(sub_eos_max)
            # print(sub_eos_max.shape)
            per_sub_max_eos_win.append(sub_eos_max[None, ...])
        for meow in per_sub_max_eos_win:
            print(meow.shape)
        per_sub_max_eos_win = np.concatenate(per_sub_max_eos_win, axis=0)
        per_sub_max_eos.append(per_sub_max_eos_win[None, ...])
        frac_sub_win = np.array(frac_sub_win)
        mean_acc_win = np.array(mean_acc_win)
        arg_max_eos_win = np.array(arg_max_eos_win)
        arg_max_tot_win = np.array(arg_max_tot_win)

        frac_sub_tot.append(frac_sub_win[None, ...])
        mean_acc_tot.append(mean_acc_win[None, ...])
        arg_max_tot.append(arg_max_tot_win[None, ...])
        arg_max_eos.append(arg_max_eos_win[None, ...])

    frac_sub_tot = np.concatenate(frac_sub_tot, axis=0)

    mean_acc_tot = np.concatenate(mean_acc_tot, axis=0)

    arg_max_tot = np.concatenate(arg_max_tot, axis=0)

    arg_max_eos = np.concatenate(arg_max_eos, axis=0)

    per_sub_max_eos = np.concatenate(per_sub_max_eos, axis=0)
    # print(mean_acc_tot[1, 2])

    fig, axs = plt.subplots(1, 2)
    h0 = axs[0].imshow(frac_sub_tot, interpolation='nearest', vmin=0.5, vmax=1.0)
    axs[0].set_xticks(range(len(num_insts)))
    axs[0].set_xticklabels(num_insts)
    axs[0].set_yticks(range(len(win_lens)))
    axs[0].set_yticklabels(np.array(win_lens).astype('float')*2)
    axs[0].set_title('Frac Subjects > Chance')
    axs[0].set_xlabel('Number of Instances')
    axs[0].set_ylabel('Window Length (ms)')
    fig.colorbar(h0, ax=axs[0], shrink=0.5)
    h1 = axs[1].imshow(mean_acc_tot, interpolation='nearest', vmin=0.25, vmax=0.5)
    axs[1].set_xticks(range(len(num_insts)))
    axs[1].set_xticklabels(num_insts)
    axs[1].set_yticks(range(len(win_lens)))
    axs[1].set_yticklabels(np.array(win_lens).astype('float')*2)
    axs[1].set_title('Mean Accuracy')
    axs[1].set_xlabel('Number of Instances')
    axs[1].set_ylabel('Window Length (ms)')
    fig.colorbar(h1, ax=axs[1], shrink=0.5)
    fig.suptitle('Post Sentence Maximum\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                             args.word,
                                                                             args.avgTime,
                                                                             args.avgTest))
    fig.tight_layout()

    plt.savefig(
        '/home/nrafidi/thesis_figs/{exp}_win_inst_comp_{sen_type}_{word}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest
        ), bbox_inches='tight')
    print(arg_max_eos)
    print(arg_max_tot)
    print(np.squeeze(per_sub_max_eos[:, 0, :]))
    plt.show()


