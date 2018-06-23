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
    # parser.add_argument('--word', default = 'noun2', choices=['noun1', 'noun2', 'verb'])
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)

    args = parser.parse_args()
    experiment = args.experiment
    subject = args.subject
    # word = args.word
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
    time_step = int(250 / 2)
    start_line = time_step
    sen_type_list = ['active'] #, 'passive']
    sen_list = [0] #, 16]
    inst_list = [10, 5, 2, 1]
    title_list = ['Single Trial', '2 Trial Average', '5 Trial Average', '10 Trial Average']
    for sen_type in sen_type_list:
        if sen_type == 'active':
            text_to_write = np.array(['A', 'dog', 'found', 'the', 'peach.'])
            max_line = 2.51 * 2 * time_step

        else:
            text_to_write = np.array(['The', 'peach', 'was', 'found', 'by', 'the', 'dog.'])
            max_line = 3.51 * 2 * time_step
        for sen_id in sen_list:
            inst_fig = plt.figure(figsize=(16, 19))
            inst_grid = AxesGrid(inst_fig, 111, nrows_ncols=(len(inst_list), 1),
                                    axes_pad=0.7, cbar_mode='single', cbar_location='right',
                                    cbar_pad=0.5, cbar_size='2%', share_all=True)
            for i_inst, num_instances in enumerate(inst_list):
                data, labels, sen_ints, time, sensor_regions = load_data.load_sentence_data_v2(subject=subject,
                                                                                               align_to='noun1',
                                                                                               voice=[sen_type],
                                                                                               experiment=experiment,
                                                                                               proc=proc,
                                                                                               num_instances=num_instances,
                                                                                               reps_filter=lambda x: [i for i in range(x) if i < 10],
                                                                                               sensor_type=None,
                                                                                               is_region_sorted=False,
                                                                                               tmin=-1.0,
                                                                                               tmax=4.5)
                # time_to_plot = range(180, 254)
                # data = data[:, :, time_to_plot]
                # time = time[time_to_plot]
                if num_instances == 1:
                    data = np.squeeze(data[sen_list[0], :, :])
                else:
                    data = np.squeeze(data[sen_id, :, :])

                data_to_plot = data[sorted_inds, ::2]
                print(np.max(data_to_plot))
                print(np.min(data_to_plot))
                time = time[::2] + 0.5
                num_time = time.size
                ax = inst_grid[i_inst]
                im = ax.imshow(data_to_plot, interpolation='nearest', vmin=-1.6e-11,
                              vmax=1.6e-11)
                ax.set_yticks(yticks_sens[1:])
                ax.set_yticklabels(uni_reg[1:])
                ax.set_xticks(range(0, num_time, time_step))
                time_labels = time[::time_step]
                time_labels[np.abs(time_labels) < 1e-10] = 0.0
                ax.set_xticklabels(['%.1f' % tm for tm in time_labels])
                ax.tick_params(labelsize=ticklabelsize)

                for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
                    ax.axvline(x=v, color='k')

                    buff_space = 0.025
                    if i_v < len(text_to_write):
                        ax.text(v + buff_space * 2 * time_step, 30, text_to_write[i_v],
                                color='k', fontsize=14)

                ax.set_title(title_list[i_inst], fontsize=axistitlesize)
                if i_inst == 3:
                    ax.set_xlabel('Time Relative to Sentence Onset (s)', fontsize=axislabelsize)
                ax.text(-0.1, 1.1, string.ascii_uppercase[i_inst], transform=ax.transAxes,
                        size=axislettersize, weight='bold')

                cbar = inst_grid.cbar_axes[0].colorbar(im)
                print(cbar)
                inst_fig.suptitle('MEG Data for {sen_type} sentence {sen_id}, Subject {subject}'.format(sen_type=sen_type, sen_id=sen_id, subject=subject),
                                   fontsize=suptitlesize)
                inst_fig.text(0.04, 0.5, 'Sensors', va='center',
                               rotation=90, rotation_mode='anchor', fontsize=axislabelsize)

                inst_fig.subplots_adjust(top=0.98)
                inst_fig.savefig(
                    '/home/nrafidi/thesis_figs/{exp}_{subject}_avg-data_{sen_type}_{sen_id}.pdf'.format(
                        subject=subject, exp=experiment, sen_type=sen_type, sen_id=sen_id
                    ), bbox_inches='tight')

                inst_fig.savefig(
                    '/home/nrafidi/thesis_figs/{exp}_{subject}_avg-data_{sen_type}_{sen_id}.png'.format(
                        subject=subject, exp=experiment, sen_type=sen_type, sen_id=sen_id
                    ), bbox_inches='tight')


    other_sub_data = load_data.load_sentence_data_v2(subject='I',
                                                       align_to='noun1',
                                                       voice=['active'],
                                                       experiment=experiment,
                                                       proc=proc,
                                                       num_instances=1,
                                                       reps_filter=lambda x: [i for i in range(x) if i < 10],
                                                       sensor_type=None,
                                                       is_region_sorted=False,
                                                       tmin=-1.0,
                                                       tmax=4.5)
    fig, ax = plt.subplots(figsize=(16, 10))
    print(type(other_sub_data))
    other_sub_data = other_sub_data[0]
    print(other_sub_data.shape)
    print(sen_id)
    other_sub_data = np.squeeze(other_sub_data[sen_id, :, :])

    data_to_plot = data_to_plot - other_sub_data[sorted_inds, ::2]
    print(np.max(data_to_plot))
    print(np.min(data_to_plot))
    time = time[::2] + 0.5
    num_time = time.size
    im = ax.imshow(data_to_plot, aspect='auto', interpolation='nearest', vmin=-1.6e-11,
                   vmax=1.6e-11)
    ax.set_yticks(yticks_sens[1:])
    ax.set_yticklabels(uni_reg[1:])
    ax.set_xticks(range(0, num_time, time_step))
    time_labels = time[::time_step]
    time_labels[np.abs(time_labels) < 1e-10] = 0.0
    ax.set_xticklabels(['%.1f' % tm for tm in time_labels])
    ax.tick_params(labelsize=ticklabelsize)

    for i_v, v in enumerate(np.arange(start_line, max_line, time_step)):
        ax.axvline(x=v, color='k')

        buff_space = 0.025
        if i_v < len(text_to_write):
            ax.text(v + buff_space * 2 * time_step, 30, text_to_write[i_v],
                    color='k', fontsize=14)


    ax.set_xlabel('Time Relative to Sentence Onset (s)', fontsize=axislabelsize)
    ax.set_ylabel('Sensors', fontsize=axislabelsize)
    fig.colorbar(im)
    fig.subplots_adjust(top=0.85)
    fig.suptitle(
        'MEG Data Difference between Subjects {subject} and I\nfor {sen_type} sentence {sen_id}'.format(sen_type=sen_type, sen_id=sen_id,
                                                                              subject=subject),
        fontsize=suptitlesize)

    fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_{subject}-I_avg-data-diff_{sen_type}_{sen_id}.pdf'.format(
            subject=subject, exp=experiment, sen_type=sen_type, sen_id=sen_id
        ), bbox_inches='tight')

    fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_{subject}-I_avg-data-diff_{sen_type}_{sen_id}.png'.format(
            subject=subject, exp=experiment, sen_type=sen_type, sen_id=sen_id
        ), bbox_inches='tight')

    plt.show()
