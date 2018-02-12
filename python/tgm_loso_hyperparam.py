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

    win_lens = [12, 25, 50, 100]
    num_insts = [1, 2, 5, 10]

    frac_sub_tot = []
    mean_acc_tot = []
    for win_len in win_lens:
        time_adjust = win_len * 0.002
        frac_sub_win = []
        mean_acc_win = []
        for num_instances in num_insts:
            intersection, acc_all, time, win_starts = tgm_loso_acc.intersect_accs(args.experiment,
                                                                     args.sen_type,
                                                                     args.word,
                                                                     win_len=win_len,
                                                                     overlap=12,
                                                                     adj=args.adj,
                                                                     num_instances=num_instances,
                                                                     avgTime=args.avgTime,
                                                                     avgTest=args.avgTest)

            time_ind = time[win_starts] >= (max_time - time_adjust)
            frac_sub = np.diag(intersection).astype('float')/float(len(load_data.VALID_SUBS[args.experiment]))
            mean_acc = np.diag(np.mean(acc_all, axis=0))

            max_frac_sub = np.max(frac_sub[time_ind])
            max_mean_acc = np.max(mean_acc[time_ind])
            frac_sub_win.append(max_frac_sub)
            mean_acc_win.append(max_mean_acc)
        frac_sub_win = np.array(frac_sub_win)
        mean_acc_win = np.array(mean_acc_win)

        frac_sub_tot.append(frac_sub_win[None, ...])
        mean_acc_tot.append(mean_acc_win[None, ...])

    frac_sub_tot = np.concatenate(frac_sub_tot, axis=0)

    mean_acc_tot = np.concatenate(mean_acc_tot, axis=0)


    print(mean_acc_tot[0, 0])

    fig, axs = plt.subplots(1, 2)
    h0 = axs[0].imshow(frac_sub_tot, interpolation='nearest')
    axs[0].set_xticks(range(len(num_insts)))
    axs[0].set_xticklabels(num_insts)
    axs[0].set_yticks(range(len(win_lens)))
    axs[0].set_yticklabels(np.array(win_lens).astype('float')*0.002)
    axs[0].set_title('Fraction of Subjects > Chance')
    axs[0].set_xlabel('Number of Instances')
    axs[0].set_ylabel('Window Length (ms)')
    fig.colorbar(h0, ax=axs[0], shrink=0.5)
    h1 = axs[1].imshow(mean_acc_tot, interpolation='nearest')
    axs[1].set_xticks(range(len(num_insts)))
    axs[1].set_xticklabels(num_insts)
    axs[1].set_yticks(range(len(win_lens)))
    axs[1].set_yticklabels(np.array(win_lens).astype('float')*0.002)
    axs[1].set_title('Mean Accuracy')
    axs[1].set_xlabel('Number of Instances')
    axs[1].set_ylabel('Window Length (ms)')
    fig.colorbar(h1, ax=axs[1], shrink=0.5)
    fig.suptitle('Post Sentence Maximum\n{} {} avgTime {} avgTest {}'.format(args.sen_type,
                                                                             args.word,
                                                                             args.avgTime,
                                                                             args.avgTest))
    fig.tight_layout()
    plt.show()


