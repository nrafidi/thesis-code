import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import run_coef_TGM_multisub
import os


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--sen_type', choices=run_coef_TGM_multisub.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['noun1', 'verb', 'noun2'])
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--alg', default='lr-l2')
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='T')
    args = parser.parse_args()

    experiment = args.experiment
    sen_type= args.sen_type
    word = args.word
    win_len = args.win_len
    overlap = args.overlap
    alg = args.alg
    adj = args.adj
    avgTime = args.avgTime
    num_instances = args.num_instances

    top_dir = run_coef_TGM_multisub.TOP_DIR.format(exp=experiment)

    fname = run_coef_TGM_multisub.SAVE_FILE.format(dir=top_dir,
                             sen_type=sen_type,
                             word=word,
                             win_len=win_len,
                             ov=overlap,
                             alg=alg,
                             adj=adj,
                             avgTm=avgTime,
                             inst=num_instances)

    result = np.load(fname + '.npz')
    maps = result['haufe_maps']
    win_starts = result['win_starts']
    time = result['time'][win_starts]
    time_to_plot = np.where(np.logical_and(time >= 0.2, time <= 0.3))

    time_to_plot = time_to_plot[0][0]
    print(time_to_plot)
    map = maps[time_to_plot]

    fig, ax = plt.subplots()
    im = ax.imshow(map, interpolation='nearest', aspect='auto')
    fig.suptitle('Raw importance map')
    fig.colorbar(im)

    class_map = np.reshape(np.mean(map, axis=0), (1, -1))
    fig, ax = plt.subplots()
    im = ax.imshow(class_map, interpolation='nearest', aspect='auto')
    fig.suptitle('Map averaged over classes')
    fig.colorbar(im)

    sub_map = np.zeros((1, 306))
    for i_sub in range(len(run_coef_TGM_multisub.VALID_SUBS[experiment])):
        start_ind = i_sub*306
        end_ind = start_ind + 306
        sub_map += class_map[:, start_ind:end_ind]
    sub_map /= len(run_coef_TGM_multisub.VALID_SUBS[experiment])

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    sub_map = sub_map[:, sorted_inds]

    fig, ax = plt.subplots(figsize=(12, 12))
    h = ax.imshow(sub_map, interpolation='nearest', aspect='auto')
    ax.set_xticks(yticks_sens)
    ax.set_xticklabels(uni_reg)
    ax.set_xlabel('Sensors')
    fig.suptitle('Map averaged over classes and Subjects')
    fig.colorbar(h)


    plt.show()


