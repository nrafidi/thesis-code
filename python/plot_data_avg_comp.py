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

TIME_LIMITS = {'PassAct3':
    {'active': {
            'noun1': {'tmin': -1.0, 'tmax': 5.0},
            'verb': {'tmin': -1.0, 'tmax': 4.5},
            'noun2': {'tmin': -2.5, 'tmax': 3.5}},
        'passive': {
            'noun1': {'tmin': -1.0, 'tmax': 5.0},
            'verb': {'tmin': -1.5, 'tmax': 4.0},
            'noun2': {'tmin': -3.5, 'tmax': 2.5}}},
'krns2': {
        'active': {
            'noun1': {'tmin': -0.5, 'tmax': 3.8},
            'verb': {'tmin': -1.0, 'tmax': 3.3},
            'noun2': {'tmin': -2.0, 'tmax': 2.3}},
        'passive': {
            'noun1': {'tmin': -0.5, 'tmax': 3.8},
            'verb': {'tmin': -1.5, 'tmax': 2.8},
            'noun2': {'tmin': -3.0, 'tmax': 1.3}}}}


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
    parser.add_argument('--proc', default=load_data.DEFAULT_PROC)

    args = parser.parse_args()
    experiment = args.experiment
    subject = args.subject
    proc = args.proc

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    stimuli_voice = list(load_data.read_stimuli(experiment))

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = np.array([sorted_reg.index(reg) for reg in uni_reg])
    sen_type_list = ['active', 'passive']
    word_list = ['noun1', 'noun2']
    for sen_type in sen_type_list:
        inst_fig = plt.figure(figsize=(16, 22))
        inst_grid = AxesGrid(inst_fig, 111, nrows_ncols=(len(word_list), 1),
                                axes_pad=0.7, cbar_mode='single', cbar_location='right',
                                cbar_pad=0.5, cbar_size='2%', share_all=True)
        word_data = []
        for i_word, word in enumerate(word_list):
            data, labels, sen_ints, time, sensor_regions = load_data.load_sentence_data_v2(subject=subject,
                                                                                           align_to=word,
                                                                                           voice=[sen_type],
                                                                                           experiment=experiment,
                                                                                           proc=proc,
                                                                                           num_instances=1,
                                                                                           reps_filter=lambda x: [i for i in range(x) if i < 10],
                                                                                           sensor_type=None,
                                                                                           is_region_sorted=False,
                                                                                           tmin=TIME_LIMITS[experiment][sen_type][word]['tmin'],
                                                                                           tmax=TIME_LIMITS[experiment][sen_type][word]['tmax'])
            print(data.shape)
            valid_inds = []
            for i_sen_int, sen_int in enumerate(sen_ints):
                sen_words = stimuli_voice[sen_int]['stimulus'].split()
                if len(sen_words) > 5:
                    valid_inds.append(i_sen_int)

            valid_inds = np.array(valid_inds)
            print(valid_inds)
            # time_to_plot = range(180, 254)
            # data = data[:, :, time_to_plot]
            # time = time[time_to_plot]
            data = data[valid_inds, ...]
            data_to_plot = np.squeeze(np.mean(data[:, sorted_inds, ::2], axis=0))
            word_data.append(data_to_plot)
            print(np.max(data_to_plot))
            print(np.min(data_to_plot))
            time = time[::2]
            num_time = time.size
            ax = inst_grid[i_word]
            im = ax.imshow(data_to_plot, interpolation='nearest', vmin=-1.6e-11,
                          vmax=1.6e-11)
            ax.set_yticks(yticks_sens[1:])
            ax.set_yticklabels(uni_reg[1:])
            ax.set_xticks(range(0, num_time, 125))
            time_labels = time[::125]
            # time_labels[np.abs(time_labels) < 1e-10] = 0.0
            ax.set_xticklabels(['%.1f' % tm for tm in time_labels])
            ax.tick_params(labelsize=ticklabelsize)
            ax.set_title(word_list[i_word], fontsize=axistitlesize)
            # if i_inst == 3:
            ax.set_xlabel('Time Relative to Sentence Onset (s)', fontsize=axislabelsize)
            ax.text(-0.1, 1.1, string.ascii_uppercase[i_word], transform=ax.transAxes,
                    size=axislettersize, weight='bold')

            cbar = inst_grid.cbar_axes[0].colorbar(im)
            print(cbar)
            inst_fig.suptitle('MEG Data for {sen_type}, Subject {subject}'.format(sen_type=sen_type, subject=subject),
                               fontsize=suptitlesize)
            inst_fig.text(0.04, 0.45, 'Sensors', va='center',
                           rotation=90, rotation_mode='anchor', fontsize=axislabelsize)

        diff_fig, diff_ax = plt.subplots()
        im = diff_ax.imshow(word_data[1] - word_data[0])
        diff_fig.suptitle(sen_type)
        diff_ax.set_yticks(yticks_sens[1:])
        diff_ax.set_yticklabels(uni_reg[1:])
        diff_ax.set_xticks(range(0, num_time, 125))
        time_labels = time[::125]
        diff_ax.set_xticklabels(['%.1f' % tm for tm in time_labels])
        diff_ax.tick_params(labelsize=ticklabelsize)
        diff_ax.set_xlabel('Time Relative to Sentence Onset (s)', fontsize=axislabelsize)
        diff_fig.colorbar(im)
            # inst_fig.subplots_adjust(top=0.85)
            # inst_fig.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_{subject}_avg-data_{sen_type}_{word}_{sen_id}.pdf'.format(
            #         subject=subject, exp=experiment, sen_type=sen_type, word=word, sen_id=sen_id
            #     ), bbox_inches='tight')
            #
            # inst_fig.savefig(
            #     '/home/nrafidi/thesis_figs/{exp}_{subject}_avg-data_{sen_type}_{word}_{sen_id}.png'.format(
            #         subject=subject, exp=experiment, sen_type=sen_type, word=word, sen_id=sen_id
            #     ), bbox_inches='tight')
    plt.show()
