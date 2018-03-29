import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
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

    if sen_type == 'active':
        max_time = 2.0
    else:
        max_time = 3.0
    time_adjust = win_len * 0.002

    if num_instances == 1:
        avgTest = 'F'
    if exp == 'krns2' and sen_type != 'active' and word != 'verb':
        rep = 10
    else:
        rep = None

    acc_by_sub = []
    acc_intersect = []
    time_by_sub = []
    win_starts_by_sub = []
    eos_max_by_sub = []
    for sub in load_data.VALID_SUBS[exp]:
        save_dir = run_TGM_LOSO.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result_fname = run_TGM_LOSO.SAVE_FILE.format(dir=save_dir,
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
                                                       rep=rep,
                                                       rsP=1,
                                                       mode='acc') + '.npz'
        if not os.path.isfile(result_fname):
            continue
        try:
            result = np.load(result_fname)
            time = np.squeeze(result['time'])
            win_starts = result['win_starts']
        except:
            print(result_fname)
            continue

        time_ind = np.where(time[win_starts] >= (max_time - time_adjust))
        time_ind = time_ind[0]

        fold_acc = result['tgm_acc']
        eos_max_fold = []
        for i_fold in range(fold_acc.shape[0]):
            diag_acc = np.diag(np.squeeze(fold_acc[i_fold, :, :]))
            # if sub =='C':
            #     fig, ax = plt.subplots()
            #     ax.imshow(np.squeeze(fold_acc[i_fold, :, :]), interpolation='nearest', aspect='auto')
            #     ax.set_title('{subject} {fold}'.format(subject=sub, fold=i_fold))
            argo = np.argmax(diag_acc[time_ind])
            eos_max_fold.append(time_ind[argo])
        eos_max_fold = np.array(eos_max_fold)
        eos_max_by_sub.append(eos_max_fold[None, :])
        acc = np.mean(fold_acc, axis=0)
        # if sub == 'B':
        #     fig, ax = plt.subplots()
        #     ax.plot(acc[:, 0])
        #     ax.set_title('B')

        time_by_sub.append(time[None, ...])
        win_starts_by_sub.append(win_starts[None, ...])
        acc_thresh = acc > 0.25
        acc_by_sub.append(acc[None, ...])
        acc_intersect.append(acc_thresh[None, ...])
    acc_all = np.concatenate(acc_by_sub, axis=0)
    intersection = np.sum(np.concatenate(acc_intersect, axis=0), axis=0)
    time = np.mean(np.concatenate(time_by_sub, axis=0), axis=0)
    win_starts = np.mean(np.concatenate(win_starts_by_sub, axis=0), axis=0).astype('int')
    eos_max = np.concatenate(eos_max_by_sub, axis=0)

    return intersection, acc_all, time, win_starts, eos_max


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

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    intersection, acc_all, time, win_starts, eos_max = intersect_accs(args.experiment,
                                                                      args.sen_type,
                                                                      args.word,
                                                                      win_len=args.win_len,
                                                                      overlap=args.overlap,
                                                                      adj=args.adj,
                                                                      num_instances=args.num_instances,
                                                                      avgTime=args.avgTime,
                                                                      avgTest=args.avgTest)

    frac_sub = np.diag(intersection).astype('float')/float(acc_all.shape[0])
    mean_acc = np.mean(acc_all, axis=0)
    
    time_step = int(250/args.overlap)
    if args.sen_type == 'active':
        text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
        max_line = 2.51 * 2 * time_step
        start_line = time_step
        if args.word == 'noun1':
            start_line -= 0.0
        elif args.word == 'verb':
            max_line -= 0.5
            start_line -= 0.5
        else:
            max_line -= 1.5
            start_line -= 1.5
    else:
        text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
        max_line = 3.51 * 2 * time_step
        start_line = time_step
        if args.word == 'noun1':
            start_line -= 0.0
        elif args.word == 'verb':
            max_line -= 1.0
            start_line -= 1.0
        else:
            max_line -= 2.5
            start_line -= 2.5
    print(mean_acc.shape)
    print(np.max(mean_acc))

    fig, ax = plt.subplots()
    h = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=0.25, vmax=0.5)
    ax.set_ylabel('Train Time (s)')
    ax.set_xlabel('Test Time (s)')
    ax.set_title('Average TGM {sen_type} {word}\nNumber of Instances: {ni} Window Size: {win}\n{aT}{aTst}'.format(ni=args.num_instances,
                                                                                                                  win=args.win_len*0.002,
                                                                                                                   aT=aT,
                                                                                                                   aTst=aTst,
                                                                                                                   sen_type=args.sen_type,
                                                                                                                   word=args.word))
    ax.set_xticks(range(0, len(time[win_starts]), time_step))
    label_time = time[win_starts]
    label_time = label_time[::time_step]
    label_time[np.abs(label_time) < 1e-15] = 0.0
    ax.set_xticklabels(label_time)
    ax.set_yticks(range(0, len(time[win_starts]), time_step))
    ax.set_yticklabels(label_time)
    time_adjust = args.win_len

    for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
        ax.axvline(x=v, color='w')
        if i_v < len(text_to_write):
            plt.text(v + 0.05 * 2*time_step, 1.5*time_step, text_to_write[i_v], color='w')
    plt.colorbar(h)
    ax.set_xlim(left=time_step)
    ax.set_ylim(top=time_step)
    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/{exp}_avg-tgm_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    fig, ax = plt.subplots()
    h = ax.imshow(np.squeeze(intersection), interpolation='nearest', aspect='auto', vmin=0, vmax=acc_all.shape[0])

    ax.set_ylabel('Train Time (s)')
    ax.set_xlabel('Test Time (s)')
    ax.set_title(
        'Intersection TGM\n{sen_type} {word} {experiment}'.format(sen_type=args.sen_type,
                                                                  word=args.word,
                                                                  experiment=args.experiment))
    ax.set_xticks(range(0, len(time[win_starts]), time_step))
    label_time = time[win_starts]
    label_time = label_time[::time_step]
    label_time[np.abs(label_time) < 1e-15] = 0.0
    ax.set_xticklabels(label_time)
    ax.set_yticks(range(0, len(time[win_starts]), time_step))
    ax.set_yticklabels(label_time)
    ax.set_xlim(left=time_step)
    ax.set_ylim(top=time_step)

    for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
        ax.axvline(x=v, color='k')
        if i_v < len(text_to_write):
            plt.text(v + 0.05 * 2*time_step, time_step, text_to_write[i_v], color='k')
    plt.colorbar(h)

    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/{exp}_intersection_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    time_adjust = args.win_len*0.002
    fig, ax = plt.subplots()

    ax.plot(np.diag(mean_acc), label='Accuracy')
    ax.plot(frac_sub, label='Fraction of Subjects > Chance')
    # if args.sen_type == 'active':
    #     text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
    #     max_line = 2.0 - time_adjust + 0.01
    #     if args.word == 'noun1':
    #         start_line = -0.5 - time_adjust
    #     else:
    #         start_line = -1.0 - time_adjust
    #
    # else:
    #     text_to_write = ['Det', 'Noun1', 'was', 'Verb', 'by', 'Det', 'Noun2.']
    #     max_line = 3.0 - time_adjust + 0.01
    #     if args.word == 'noun1':
    #         start_line = -0.5 - time_adjust
    #     else:
    #         start_line = -1.5 - time_adjust

    ax.set_xticks(range(0, len(time[win_starts]), time_step))
    label_time = time[win_starts]
    label_time = label_time[::time_step]
    label_time[np.abs(label_time) < 1e-15] = 0.0
    ax.set_xticklabels(label_time)
    for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
        ax.axvline(x=v, color='k')
        if i_v < len(text_to_write):
            plt.text(v + 0.05, 0.8, text_to_write[i_v])
    ax.set_ylabel('Accuracy/Fraction > Chance')
    ax.set_xlabel('Time')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim(left=time_step)
    ax.legend(loc=4)
    ax.set_title('Mean Acc over subjects and Frac > Chance\n{sen_type} {word} {experiment}'.format(sen_type=args.sen_type,
                                                                                 word=args.word,
                                                                                 experiment=args.experiment))

    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/{exp}_diag_acc_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')


    plt.show()


