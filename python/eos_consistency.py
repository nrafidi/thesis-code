import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import run_TGM_LOSO
import tgm_loso_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_TGM_LOSO.VALID_SEN_TYPE)
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='F')
    args = parser.parse_args()

    word_list = ['noun1', 'verb', 'noun2']
    eos_max_by_word = []
    # time_by_word = []
    for word in word_list:
        _, _, time, win_starts, eos_max = tgm_loso_acc.intersect_accs(args.experiment,
                                                                      args.sen_type,
                                                                      word,
                                                                      win_len=args.win_len,
                                                                      overlap=args.overlap,
                                                                      adj=args.adj,
                                                                      num_instances=args.num_instances,
                                                                      avgTime=args.avgTime,
                                                                      avgTest=args.avgTest)
        time_to_use = time[win_starts]
        # time_by_word.append[time_to_use[None, ...]]
        eos_max_by_word.append(eos_max[None, ...])

    # time_by_word = np.concatenate(time_by_word)
    eos_max_by_word = np.concatenate(eos_max_by_word)*0.025

    mean_eos_max_by_word = np.mean(eos_max_by_word, axis=2)
    std_eos_max_by_word = np.std(eos_max_by_word, axis=2)

    [num_words, num_sub] = mean_eos_max_by_word.shape
    ind = np.arange(num_sub)
    width = 0.27
    colors = ['r', 'g', 'b', 'c', 'k', 'm']
    fig, ax = plt.subplots()
    for i_word in range(num_words):
        ax.bar(ind + width*i_word, mean_eos_max_by_word[i_word, :], width, color=colors[i_word], yerr=std_eos_max_by_word[i_word, :], label=word_list[i_word])
    ax.set_ylabel('Time of EOS Max')
    ax.set_ylim(bottom=2.0)
    ax.set_title('Consistency of EOS Max Time')
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(load_data.VALID_SUBS[args.experiment])
    ax.legend(loc=4)

    plt.show()
