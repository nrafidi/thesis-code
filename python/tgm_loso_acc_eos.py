import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import run_TGM_LOSO_EOS


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'
CHANCE = {'pooled': {'noun1': 0.125,
                     'verb': 0.25,
                     'voice': 0.5},
          'active': {'noun1': 0.25,
                     'verb': 0.25,
                     'voice': 0.5},
          'passive': {'noun1': 0.25,
                     'verb': 0.25,
                     'voice': 0.5}
          }

def intersect_accs(exp,
                   sen_type,
                   word,
                   win_len=100,
                   overlap=12,
                   adj=None,
                   num_instances=1,
                   avgTime='F',
                   avgTest='F'):
    top_dir = run_TGM_LOSO_EOS.TOP_DIR.format(exp=exp)

    chance=CHANCE[sen_type][word]

    if num_instances == 1:
        avgTest = 'F'

    acc_by_sub = []
    acc_intersect = []
    time_by_sub = []
    win_starts_by_sub = []
    eos_max_by_sub = []
    for sub in load_data.VALID_SUBS[exp]:
        save_dir = run_TGM_LOSO_EOS.SAVE_DIR.format(top_dir=top_dir, sub=sub)
        result_fname = run_TGM_LOSO_EOS.SAVE_FILE.format(dir=save_dir,
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
                                                         rsP=1,
                                                         mode='acc') + '.npz'
        if not os.path.isfile(result_fname):
            print(result_fname)
            continue
        result = np.load(result_fname)
        time = np.squeeze(result['time'])
        win_starts = result['win_starts']

        fold_acc = result['tgm_acc']
        eos_max_fold = []
        for i_fold in range(fold_acc.shape[0]):
            diag_acc = np.diag(np.squeeze(fold_acc[i_fold, :, :]))
            # if sub =='C':
            #     fig, ax = plt.subplots()
            #     ax.imshow(np.squeeze(fold_acc[i_fold, :, :]), interpolation='nearest', aspect='auto')
            #     ax.set_title('{subject} {fold}'.format(subject=sub, fold=i_fold))
            argo = np.argmax(diag_acc)
            eos_max_fold.append(argo)
        eos_max_fold = np.array(eos_max_fold)
        eos_max_by_sub.append(eos_max_fold[None, :])
        acc = np.mean(fold_acc, axis=0)
        # if sub == 'B':
        #     fig, ax = plt.subplots()
        #     ax.plot(acc[:, 0])
        #     ax.set_title('B')

        time_by_sub.append(time[None, ...])
        win_starts_by_sub.append(win_starts[None, ...])
        acc_thresh = acc > chance
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
    parser.add_argument('--sen_type', choices=run_TGM_LOSO_EOS.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['noun1', 'verb', 'voice'])
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='F')
    args = parser.parse_args()

    chance = CHANCE[args.sen_type][args.word]

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
    print(mean_acc.shape)
    print(np.max(mean_acc))
    # for sub in range(acc_all.shape[0]):
    #     fig, ax = plt.subplots()
    #     h = ax.imshow(np.squeeze(acc_all[sub, ...]), interpolation='nearest', aspect='auto', vmin=0, vmax=1.0)
    #     ax.set_ylabel('Test Time')
    #     ax.set_xlabel('Train Time')
    #     ax.set_title('Subject {sub} TGM\n{sen_type} {word} {experiment}'.format(sub=load_data.VALID_SUBS[args.experiment][sub],
    #                                                                             sen_type=args.sen_type,
    #                                                                             word=args.word,
    #                                                                             experiment=args.experiment))
    #     ax.set_xticks(range(0, len(time[win_starts]), time_step))
    #     label_time = time[win_starts]
    #     label_time = label_time[::time_step]
    #     label_time[np.abs(label_time) < 1e-15] = 0.0
    #     ax.set_xticklabels(label_time)
    #     ax.set_yticks(range(0, len(time[win_starts]), time_step))
    #     ax.set_yticklabels(label_time)
    #     time_adjust = args.win_len
    #
    #     plt.colorbar(h)

    fig, ax = plt.subplots()
    h = ax.imshow(np.squeeze(intersection), interpolation='nearest', aspect='auto', vmin=0, vmax=acc_all.shape[0])
    ax.set_ylabel('Test Time')
    ax.set_xlabel('Train Time')
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

    plt.colorbar(h)

    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_intersection_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    fig, ax = plt.subplots()
    h = ax.imshow(np.squeeze(mean_acc), interpolation='nearest', aspect='auto', vmin=chance, vmax=2*chance)
    ax.set_ylabel('Train Time (s)')
    ax.set_xlabel('Test Time (s)')
    ax.set_title('Average TGM {sen_type} {word}\nNumber of Instances: {ni} Window Size: {win}\n{aT}{aTst}'.format(
        ni=args.num_instances,
        win=args.win_len * 0.002,
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
    ax.axvline(x=0.625*time_step, color='w')
    plt.colorbar(h)

    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_avgtgm_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

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

    ax.set_ylabel('Accuracy/Fraction > Chance')
    ax.set_xlabel('Time')
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc=4)
    ax.set_title('Mean Acc over subjects and Frac > Chance\n{sen_type} {word} {experiment}'.format(sen_type=args.sen_type,
                                                                                 word=args.word,
                                                                                 experiment=args.experiment))

    fig.tight_layout()
    plt.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_diag_acc_{sen_type}_{word}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, sen_type=args.sen_type, word=args.word, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')


    plt.show()


