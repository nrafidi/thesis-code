import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from syntax_vs_semantics import load_data
import models
import numpy as np
import os.path
from mpl_toolkits.axes_grid1 import AxesGrid
import scipy.io as sio
import string

NUM_REPS = {'krns2': 15, 'PassAct2': 10, 'PassAct3': 10}
VALID_SUBS = {'krns2': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
              'PassAct2': ['A', 'B', 'C'],
              'PassAct3': ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']}

VALID_SEN_TYPE = ['active', 'passive']

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
    parser.add_argument('--experiment', default='PassAct3')
    parser.add_argument('--subject', default='Z')
    parser.add_argument('--word', default = 'noun2', choices=['noun1', 'noun2', 'verb'])
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)

    args = parser.parse_args()
    experiment = args.experiment
    subject = args.subject
    word = args.word
    proc = args.proc

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = np.array([sorted_reg.index(reg) for reg in uni_reg])
    sen_type_list = ['active', 'passive']
    sen_list = [0, 8]
    inst_list = [1]
    title_list = ['10 Trial Average', '2 Trial Average', 'Single Trial', '5 Trial Average']
    for sen_type in sen_type_list:
        for sen_id in sen_list:
            inst_fig = plt.figure(figsize=(16, 22))
            inst_grid = AxesGrid(inst_fig, 111, nrows_ncols=(len(inst_list), 1),
                                    axes_pad=0.7, cbar_mode='single', cbar_location='right',
                                    cbar_pad=0.5, cbar_size='2%', share_all=True)
            for i_inst, num_instances in enumerate(inst_list):
                data, labels, sen_ints, time, sensor_regions = load_data.load_sentence_data_v2(subject=subject,
                                                                                               align_to=word,
                                                                                               voice=[sen_type],
                                                                                               experiment=experiment,
                                                                                               proc=proc,
                                                                                               num_instances=num_instances,
                                                                                               reps_filter=lambda x: [i for i in range(x) if i < 10],
                                                                                               sensor_type=None,
                                                                                               is_region_sorted=False,
                                                                                               tmin=-4.0,
                                                                                               tmax=3.0)
                # time_to_plot = time < 4.0
                # data = data[:, :, time_to_plot]
                # time = time[time_to_plot]
                if num_instances == 1:
                    data_to_plot = np.squeeze(data[sen_list[0], sorted_inds, ::2])
                else:
                    data_to_plot = np.squeeze(data[sen_id, sorted_inds, ::2])
                print(np.max(data_to_plot))
                print(np.min(data_to_plot))
                time = time[::2]
                num_time = time.size
                ax = inst_grid[i_inst]
                im = ax.imshow(data_to_plot, interpolation='nearest', vmin=-1.6e-11,
                              vmax=1.6e-11)
                ax.set_yticks(yticks_sens[1:])
                ax.set_yticklabels(uni_reg[1:])
                ax.set_xticks(range(0, num_time, 125))
                time_labels = time[::125] + 0.5
                time_labels[np.abs(time_labels) < 1e-10] = 0.0
                ax.set_xticklabels(time_labels)
                ax.tick_params(labelsize=ticklabelsize)
                ax.set_title(title_list[i_inst], fontsize=axistitlesize)
                if i_inst == 3:
                    ax.set_xlabel('Time Relative to Sentence Onset (s)', fontsize=axislabelsize)
                ax.text(-0.1, 1.1, string.ascii_uppercase[i_inst], transform=ax.transAxes,
                        size=axislettersize, weight='bold')

            cbar = inst_grid.cbar_axes[0].colorbar(im)
            print(cbar)
            inst_fig.suptitle('MEG Data for {sen_type} sentence {sen_id}, Subject {subject}'.format(sen_type=sen_type, sen_id=sen_id, subject=subject),
                               fontsize=suptitlesize)
            inst_fig.text(0.04, 0.45, 'Sensors', va='center',
                           rotation=90, rotation_mode='anchor', fontsize=axislabelsize)

            # inst_fig.subplots_adjust(top=0.85)
            inst_fig.savefig(
                '/home/nrafidi/thesis_figs/{exp}_{subject}_avg-data_{sen_type}_{word}_{sen_id}.pdf'.format(
                    subject=subject, exp=experiment, sen_type=sen_type, word=word, sen_id=sen_id
                ), bbox_inches='tight')

            inst_fig.savefig(
                '/home/nrafidi/thesis_figs/{exp}_{subject}_avg-data_{sen_type}_{word}_{sen_id}.png'.format(
                    subject=subject, exp=experiment, sen_type=sen_type, word=word, sen_id=sen_id
                ), bbox_inches='tight')
    plt.show()
