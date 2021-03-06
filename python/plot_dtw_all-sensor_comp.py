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

    dist_list = ['cosine', 'euclidean']
    inst_list = [2, 10]
    tmin_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    tlen_list = [0.05, 0.1, 0.5]
    metric_list = ['dtw', 'total']

    # How does averaging help?
    combo_fig, combo_grid = plt.subplots(1, 3)
    for i_inst, inst in enumerate(inst_list):
        total_rdm, comp_rdm = load_rdm(exp, sub, inst, voice, tmin_list[1],tmin_list[1] + tlen_list[0], dist_list[0], radius, metric_list[0])
        total_rdm /= np.max(total_rdm)
        ax = combo_grid[i_inst]
        score, _ = ktau_rdms(total_rdm, comp_rdm)
        im = ax.imshow(total_rdm, interpolation='none', vmin=0.0, vmax=1.0) #, aspect='auto')
        score_str = 'Score: %.2f' % score
        ax.set_title('{} Instances\n'.format(inst) + score_str)
    ax = combo_grid[-1]
    im = ax.imshow(comp_rdm, interpolation='none',  vmin=0.0, vmax=1.0) #aspect='auto')
    ax.set_title('Ideal')
    combo_fig.subplots_adjust(right=0.8)
    cbar_ax = combo_fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im, cax=cbar_ax)
    combo_fig.suptitle('Averaging Comparison')


    # How does this evolve over time?
    combo_fig = plt.figure(figsize=(12, 20))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(len(tmin_list), len(tlen_list)),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, aspect=True)
    i_grid = 0
    for i_tmin, tmin in enumerate(tmin_list):
        for j_tlen, tlen in enumerate(tlen_list):
            tmax = tmin + tlen
            total_rdm, comp_rdm = load_rdm(exp, sub, inst_list[0], voice, tmin, tmax,
                                           dist_list[0], radius, metric_list[0])
            total_rdm /= np.max(total_rdm)
            score, _ = ktau_rdms(total_rdm, comp_rdm)
            score_str = 'Score: %.2f' % score
            ax = combo_grid[i_grid]
            im = ax.imshow(total_rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
            ax.set_title('%.3f-%.3f\n' % (tmin, tmax) + score_str)
            i_grid += 1

    cbar = combo_grid.cbar_axes[0].colorbar(im)

    # Does dtw help?
    combo_fig = plt.figure(figsize=(20, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(1, 3),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, aspect=True)
    i_grid = 0
    for i_grid, metric in enumerate(metric_list):

        total_rdm, comp_rdm = load_rdm(exp, sub, inst_list[0], voice, tmin_list[1],tmin_list[1] + tlen_list[0],
                                   dist_list[0], radius, metric)
        total_rdm /= np.max(total_rdm)
        score, _ = ktau_rdms(total_rdm, comp_rdm)
        score_str = 'Score: %.2f' % score
        ax = combo_grid[i_grid]
        im = ax.imshow(total_rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_title(metric + '\n' + score_str)
    ax = combo_grid[-1]
    im = ax.imshow(comp_rdm, interpolation='none', vmin=0.0, vmax=1.0)  # aspect='auto')
    ax.set_title('Ideal')
    cbar = combo_grid.cbar_axes[0].colorbar(im)

    # Euclidean vs cosine?
    combo_fig = plt.figure(figsize=(20, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(1, 3),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, aspect=True)
    i_grid = 0
    for i_grid, dist in enumerate(dist_list):
        total_rdm, comp_rdm = load_rdm(exp, sub, inst_list[0], voice, tmin_list[1], tmin_list[1] + tlen_list[0],
                                       dist, radius, metric_list[0])
        total_rdm /= np.max(total_rdm)
        score, _ = ktau_rdms(total_rdm, comp_rdm)
        score_str = 'Score: %.2f' % score
        ax = combo_grid[i_grid]
        im = ax.imshow(total_rdm, interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_title(dist + '\n' + score_str)
    ax = combo_grid[-1]
    im = ax.imshow(comp_rdm, interpolation='none', vmin=0.0, vmax=1.0)  # aspect='auto')
    ax.set_title('Ideal')
    cbar = combo_grid.cbar_axes[0].colorbar(im)



    plt.show()
