import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import os
from scipy.stats import spearmanr, kendalltau
from mpl_toolkits.axes_grid1 import AxesGrid

RESULT_FNAME = '/share/volume0/nrafidi/DTW/EOS_{metric}_sensor{i_sensor}_score_{exp}_{sub}_sen{sen0}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}_{voice}.npz'
SCORE_FNAME = '/share/volume0/nrafidi/DTW/EOS_{metric}_{sensor}_corr_{exp}_{sub}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}_{voice}.npz'

NUM_SEN = {'active': 16, 'passive': 16, 'pooled': 32}


def ktau_rdms(rdm1, rdm2):
    # from Mariya Toneva
    diagonal_offset = -1 # exclude the main diagonal
    upper_tri_inds = np.triu_indices(rdm1.shape[0], diagonal_offset)
    rdm_kendall_tau, rdm_kendall_tau_pvalue = kendalltau(rdm1[upper_tri_inds],rdm2[upper_tri_inds])
    return rdm_kendall_tau, rdm_kendall_tau_pvalue


def load_rdm(exp, sub, num_instances, voice, tmin, tmax, dist, radius, metric):
    num_sen = NUM_SEN[voice]
    comp_rdm = np.ones((num_instances * num_sen, num_instances * num_sen), dtype=float)
    total_rdm = np.empty((num_instances * num_sen, num_instances * num_sen), dtype='float')
    for sen0 in range(num_sen):
        start_ind = sen0 * num_instances
        end_ind = start_ind + num_instances
        comp_rdm[start_ind:end_ind, start_ind:end_ind] = 0.0

        result_fname = RESULT_FNAME.format(exp=exp, sub=sub, sen0=sen0, radius=radius, dist=dist, voice=voice,
                                           ni=num_instances, tmin=tmin, tmax=tmax, i_sensor=-1, metric=metric)
        if not os.path.isfile(result_fname):
            print(result_fname)
        result = np.load(result_fname)
        dtw_part = result['dtw_part']
        other_sens = range(dtw_part.shape[0])
        for sen1 in other_sens:
            start_ind_y = start_ind + sen1 * num_instances
            end_ind_y = start_ind_y + num_instances
            total_rdm[start_ind:end_ind, start_ind_y:end_ind_y] = dtw_part[sen1, :, :]
            total_rdm[start_ind_y:end_ind_y, start_ind:end_ind] = dtw_part[sen1, :, :]
    return total_rdm, comp_rdm


def pad_array(arr, desired_size):
    curr_size =arr.shape
    new_arr = -1.0 * np.ones(desired_size, dtype=float)

    start_x = desired_size[0] - curr_size[0]
    start_y = desired_size[1] - curr_size[1]

    new_arr[start_x:(start_x + curr_size[0]), start_y:(start_y + curr_size[1])] = arr
    return new_arr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--voice', default='active', choices=['active', 'passive', 'pooled'])

    args = parser.parse_args()

    exp = args.experiment
    sub = args.subject
    radius = args.radius
    voice = args.voice

    dist_list = ['euclidean', 'cosine']
    inst_list = [2, 10]
    tmin_list = [0.0, 0.1, 0.2, 0.3]
    tlen_list = [0.05, 0.1, 0.5]
    metric_list = ['dtw', 'total']

    # How does averaging help?
    combo_fig = plt.figure(figsize=(20, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(1, 3),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, aspect=False)
    for i_inst, inst in enumerate(inst_list):
        total_rdm, comp_rdm = load_rdm(exp, sub, inst, voice, tmin_list[1],tmin_list[1] + tlen_list[1], dist_list[1], radius, metric_list[0])
        total_rdm /= np.max(total_rdm)
        ax = combo_grid[i_inst]
        if inst == 2:
            total_rdm = pad_array(total_rdm, (160, 160))
        im = ax.imshow(total_rdm, interpolation='nearest', vmin=0.0, vmax=1.0) #, aspect='auto')
        ax.set_title('{} Instances'.format(inst))
    ax = combo_grid[-1]
    im = ax.imshow(comp_rdm, interpolation='nearest',  vmin=0.0, vmax=1.0) #aspect='auto')
    ax.set_title('Ideal')

    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('Averaging Comparison')

    plt.show()
