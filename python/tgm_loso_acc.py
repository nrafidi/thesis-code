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
        coef_time = np.all(coef_time, axis=0)
        coef_by_sub.append(coef_time[None, ...])
    intersection = np.all(np.concatenate(coef_by_sub, axis=0), axis=0)
    return intersection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_TGM_LOSO.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['noun1', 'verb', 'noun2'])
    parser.add_argument('--win_time', type=int)
    parser.add_argument('--win_len', type=int, default=100)
    parser.add_argument('--overlap', type=int, default=12)
    parser.add_argument('--adj', default='None', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='F')
    parser.add_argument('--avgTest', default='F')
    args = parser.parse_args()

    intersection = intersect_coef(args.experiment,
                                  args.sen_type,
                                  args.word,
                                  args.win_time,
                                  win_len=args.win_len,
                                  overlap=args.overlap,
                                  adj=args.adj,
                                  num_instances=args.num_instances,
                                  avgTime=args.avgTime,
                                  avgTest=args.avgTest)

    intersection = np.reshape(intersection, (306, args.win_len))

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    intersection = intersection[sorted_inds, :]

    fig, ax = plt.subplots()
    h = ax.imshow(intersection, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xlabel('Time')
    ax.set_title('Intersection over subjects at time window {win_time}\n{sen_type} {word} {experiment}'.format(win_time=args.win_time,
                                                                                                               sen_type=args.sen_type,
                                                                                                               word=args.word,
                                                                                                               experiment=args.experiment))
    plt.colorbar(h)
    plt.show()


