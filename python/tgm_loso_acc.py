import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import run_TGM_LOSO


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


def intersect_accs(exp,
                   sen_type,
                   word,
                   win_len=100,
                   overlap=12,
                   adj=None,
                   num_instances=1,
                   avgTime='F',
                   avgTest='F'):
    top_dir = run_TGM_LOSO.TOP_DIR.format(exp=exp)

    acc_by_sub = []
    acc_intersect = []
    time_by_sub = []
    win_starts_by_sub = []
    for sub in load_data.VALID_SUBS[exp]:
        save_dir = run_TGM_LOSO.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result = np.load(run_TGM_LOSO.SAVE_FILE.format(dir=save_dir,
                                                       sub=sub,
                                                       sen_type=sen_type,
                                                       word=word,
                                                       win_len=win_len,
                                                       ov=overlap,
                                                       perm='F',
                                                       alg='lr-l1',
                                                       adj=adj,
                                                       avgTm=avgTime,
                                                       avgTst=avgTest,
                                                       inst=num_instances,
                                                       rep=10,
                                                       rsP=1,
                                                       mode='acc') + '.npz')

        acc = np.mean(result['tgm_acc'], axis=0)
        time_by_sub.append(result['time'][None, ...])
        win_starts_by_sub.append(result['win_starts'][None, ...])
        acc_thresh = acc > 0.25
        acc_by_sub.append(acc[None, ...])
        acc_intersect.append(acc_thresh[None, ...])
    acc_all = np.concatenate(acc_by_sub, axis=0)
    intersection = np.sum(np.concatenate(acc_intersect, axis=0), axis=0)
    time = np.mean(np.concatenate(time_by_sub, axis=0), axis=0)
    win_starts = np.mean(np.concatenate(win_starts_by_sub, axis=0), axis=0).astype('int')
    return intersection, acc_all, time, win_starts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_TGM_LOSO.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['noun1', 'verb', 'noun2'])
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='F')
    args = parser.parse_args()

    intersection, acc_all, time, win_starts = intersect_accs(args.experiment,
                                                             args.sen_type,
                                                             args.word,
                                                             win_len=args.win_len,
                                                             overlap=args.overlap,
                                                             adj=args.adj,
                                                             num_instances=args.num_instances,
                                                             avgTime=args.avgTime,
                                                             avgTest=args.avgTest)

    frac_sub = np.diag(intersection).astype('float')/float(len(load_data.VALID_SUBS[args.experiment]))
    mean_acc = np.mean(acc_all, axis=0)

    # fig, ax = plt.subplots()
    # h = ax.imshow(intersection, interpolation='nearest', aspect='auto', vmin=0, vmax=len(load_data.VALID_SUBS[args.experiment]))
    # ax.set_ylabel('Test Time')
    # ax.set_xlabel('Train Time')
    # ax.set_title('Intersection of acc > chance over subjects\n{sen_type} {word} {experiment}'.format(sen_type=args.sen_type,
    #                                                                                                   word=args.word,
    #                                                                                                   experiment=args.experiment))
    # plt.colorbar(h)

    # fig, ax = plt.subplots()
    # ax.plot(np.diag(intersection))
    # ax.set_ylabel('Number of Subjects with > chance acc')
    # ax.set_xlabel('Time')
    # ax.set_title('Intersection of acc > chance over subjects\n{sen_type} {word} {experiment}'.format(sen_type=args.sen_type,
    #                                                                              word=args.word,
    #                                                                              experiment=args.experiment))

    # fig, ax = plt.subplots()
    # h = ax.imshow(mean_acc, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    # ax.set_ylabel('Test Time')
    # ax.set_xlabel('Train Time')
    # ax.set_title('Mean Acc TGM over subjects\n{sen_type} {word} {experiment}'.format(sen_type=args.sen_type,
    #                                                                                  word=args.word,
    #                                                                                  experiment=args.experiment))
    # plt.colorbar(h)

    time_adjust = args.win_len*0.002
    fig, ax = plt.subplots()
    ax.plot(time[win_starts], np.diag(mean_acc), label='Accuracy')
    ax.plot(time[win_starts], frac_sub, label='Fraction of Subjects > Chance')
    if args.sen_type == 'active':
        text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
        max_line = 2.0 - time_adjust
    else:
        text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
        max_line = 3.0 - time_adjust

    for i_v, v in enumerate(np.arange(-0.5 - time_adjust, max_line, 0.5)):
        ax.vline(x=v, color='k')
        if i_v < len(text_to_write):
            plt.text(v + 0.05, 0.8, text_to_write[i_v])
    ax.set_ylabel('Accuracy/Fraction > Chance')
    ax.set_xlabel('Time')
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc=4)
    ax.set_title('Mean Acc over subjects and Frac > Chance\n{sen_type} {word} {experiment}'.format(sen_type=args.sen_type,
                                                                                 word=args.word,
                                                                                 experiment=args.experiment))


    plt.show()


