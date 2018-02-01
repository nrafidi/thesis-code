import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import run_TGM_LOSO


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


def intersect_coef(exp,
                   sen_type,
                   word,
                   win_time,
                   win_len=100,
                   overlap=12,
                   adj=None,
                   num_instances=1,
                   avgTime='F',
                   avgTest='F'):
    top_dir = run_TGM_LOSO.TOP_DIR.format(exp=exp)

    coef_by_sub = []
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
                                                       mode='coef') + '.npz')
        coef = result['coef']
        Cs = result['Cs']
        coef_time = np.array(coef[win_time] != 0)
        C_time = np.array(Cs[win_time])
        print(C_time)
        print(np.sum(coef_time))
        coef_time = np.sum(coef_time, axis=0)
        coef_by_sub.append(coef_time[None, ...])
    intersection = np.sum(np.concatenate(coef_by_sub, axis=0), axis=0)
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
    h = ax.imshow(intersection, interpolation='nearest', aspect='auto', vmin=0, vmax=len(load_data.VALID_SUBS[args.experiment]))
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


