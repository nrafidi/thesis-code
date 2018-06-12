import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import run_coef_TGM_multisub
from mne.viz import plot_topomap
from syntax_vs_semantics import sensor_metadata
from mpl_toolkits.axes_grid1 import AxesGrid


# SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

#
# def sort_sensors():
#     load_var = sio.loadmat(SENSOR_MAP)
#     sensor_reg = load_var['sensor_reg']
#     sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
#     sorted_inds = np.argsort(sensor_reg)
#     sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
#     return sorted_inds, sorted_reg


def _make_coordinates(sensor_layout, sensor_type_id):
    coordinates = np.hstack(
        [np.expand_dims(np.array([si.x_coord for si in sensor_layout if si.sensor_type_id == sensor_type_id]), 1),
         np.expand_dims(np.array([si.y_coord for si in sensor_layout if si.sensor_type_id == sensor_type_id]), 1)])
    # coordinates is now sensors by xy
    return coordinates


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
    parser.add_argument('--sensor_type_id', type=int, default=2)
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
    sensor_id = args.sensor_type_id

    tmin_to_plot = 0.0
    tmax_to_plot = 0.3
    t_step = overlap*0.002

    sensor_layout = sensor_metadata.read_sensor_metadata()
    coordinates = _make_coordinates(sensor_layout, sensor_id)
    sensors_to_keep = [i_s for i_s, s in enumerate(sensor_layout) if s.sensor_type_id == sensor_id]

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


    time_to_plot = np.arange(tmin_to_plot, tmax_to_plot, t_step).tolist()
    n_rows = 5
    n_cols = len(time_to_plot)/n_rows
    combo_fig = plt.figure(figsize=(n_cols * 6, n_rows * 6))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(n_rows, n_cols),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.2)
    for i_t, tmin in enumerate(time_to_plot):
        time_to_plot = np.where(np.logical_and(time >= tmin, time <= tmin + t_step))
        time_to_plot = time_to_plot[0][0]
        print(time_to_plot)
        map = maps[time_to_plot]
        class_map = np.mean(map, axis=0)
        sub_map = np.zeros((306,))
        for i_sub in range(len(run_coef_TGM_multisub.VALID_SUBS[experiment])):
            start_ind = i_sub*306
            end_ind = start_ind + 306
            sub_map += class_map[start_ind:end_ind]
        sub_map /= len(run_coef_TGM_multisub.VALID_SUBS[experiment])
        sub_map = sub_map[sensors_to_keep]
        print(np.min(sub_map))
        print(np.max(sub_map))
        ax = combo_grid[i_t]
        im, _ = plot_topomap(sub_map, coordinates, axes=ax, show=False)
        ax.set_title('%.2f-%.2f' % (tmin, tmin + win_len*0.002))

    cbar = combo_grid.cbar_axes[i_t].colorbar(im)

    plt.show()


