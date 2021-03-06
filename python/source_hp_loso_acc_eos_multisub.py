import mne
import numpy as np
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Adapted from Mariya Toneva
REGIONS = ['superiorfrontal', 'caudalmiddlefrontal', 'rostralmiddlefrontal', 'parsopercularis', 'parsorbitalis',
              'parstriangularis', 'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole', 'paracentral',
              'precentral', 'insula', 'postcentral', 'inferiorparietal', 'supramarginal', 'superiorparietal',
              'precuneus', 'cuneus', 'lateraloccipital', 'lingual', 'pericalcarine', 'isthmuscingulate',
              'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate', 'entorhinal',
              'parahippocampal', 'temporalpole', 'fusiform', 'superiortemporal', 'inferiortemporal', 'middletemporal',
              'transversetemporal', 'bankssts']

HEMIS = ['lh', 'rh']

TOP_DIR = '/share/volume0/nrafidi/PassAct3_TGM_LOSO_EOS_SOURCE/'
MULTI_SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_{reg}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{rank_str}{mode}'

STRUCTURAL = 'PassAct3_T'

SUBJ_DIR = '/bigbrain/bigbrain.usr1/meg/structural'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sen_type', choices=run_TGM_LOSO_EOS.VALID_SEN_TYPE)
    parser.add_argument('--word', choices = ['verb', 'voice', 'agent', 'patient', 'propid', 'bind'])
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='T')
    parser.add_argument('--avgTest', default='T')
    parser.add_argument('--accThresh', type=float, default=0.5)
    parser.add_argument('--plot_type', default='mean', choices=['max', 'mean'])
    parser.add_argument('--tmin', default=0.0, type=float)
    parser.add_argument('--tmax', default=0.5, type=float)
    parser.add_argument('--surface', default='inflated') #, choices=['pial', 'inflated'])
    args = parser.parse_args()

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    print(mne.__version__)
    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    sen_type = 'pooled'
    experiment = 'PassAct3'
    word = args.word
    acc_thresh=args.accThresh

    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    acc_thresh_str = '%.2f' % acc_thresh
    time_win_str = '%.2f-%.2f' % (args.tmin, args.tmax)

    source_by_time_mat = {}
    for hemi in HEMIS:
        hemi_mat = []
        for reg in REGIONS:
            multi_file = MULTI_SAVE_FILE.format(dir=TOP_DIR,
                                                sen_type=sen_type,
                                                word=word,
                                                reg='{}-{}'.format(reg, hemi),
                                                win_len=args.win_len,
                                                ov=args.overlap,
                                                perm='F',
                                                alg=args.alg,
                                                adj=args.adj,
                                                avgTm=args.avgTime,
                                                avgTst=args.avgTest,
                                                inst=args.num_instances,
                                                rsP=1,
                                                rank_str='',
                                                mode='acc')
            rank_file = MULTI_SAVE_FILE.format(dir=TOP_DIR,
                                               sen_type=sen_type,
                                               word=word,
                                               reg='{}-{}'.format(reg, hemi),
                                               win_len=args.win_len,
                                               ov=args.overlap,
                                               perm='F',
                                               alg=args.alg,
                                               adj=args.adj,
                                               avgTm=args.avgTime,
                                               avgTst=args.avgTest,
                                               inst=args.num_instances,
                                               rsP=1,
                                               rank_str='rank',
                                               mode='acc')

            result = np.load(multi_file + '.npz')
            rank_result = np.load(rank_file + '.npz')
            multi_fold_acc = rank_result['tgm_rank']

            time = result['time']
            win_starts = result['win_starts']
            time_win = time[win_starts]
            min_time = 0.0
            max_time = 0.5 * len(time_win) / time_step
            label_time = np.arange(min_time, max_time, 0.01)
            # print(min_time)
            # print(max_time)
            # print(len(label_time))
            time_to_plot = np.logical_and(label_time >= args.tmin, label_time < args.tmax)

            mean_acc = np.mean(multi_fold_acc, axis=0)
            diag_acc = np.diag(mean_acc)
            # print(len(diag_acc))
            diag_acc_win = diag_acc[time_to_plot]
            hemi_mat.append(diag_acc_win[None, ...])
        source_by_time_mat[hemi] = np.concatenate(hemi_mat, axis=0)


    if args.plot_type == 'mean':
        max_over_time_left = np.nanmean(source_by_time_mat['lh'], axis=1)
        max_over_time_right = np.nanmean(source_by_time_mat['rh'], axis=1)
    else:
        max_over_time_left = np.nanmax(source_by_time_mat['lh'], axis=1)
        max_over_time_right = np.nanmax(source_by_time_mat['rh'], axis=1)


    # the left and right data and vertices are required to be in different arrays for the plotting.. okay
    label_left = mne.read_labels_from_annot(
        STRUCTURAL, parc='aparc', subjects_dir=SUBJ_DIR, hemi='lh',
        verbose=False)
    label_right = mne.read_labels_from_annot(
        STRUCTURAL, parc='aparc', subjects_dir=SUBJ_DIR, hemi='rh',
        verbose=False)


    left_vertices_per_label = []
    left_maxes_per_label = []
    right_vertices_per_label = []
    right_maxes_per_label = []
    for label in label_left:
        label_name = label.name[:-3]
        # print(label_name)
        if label_name not in REGIONS:
            continue
        region_index_in_results = REGIONS.index(label_name)
        if max_over_time_left[region_index_in_results] >= acc_thresh:  # include this regiion in the plotting
            # print(type(label.vertices))
            # print(type(label.vertices[0]))

            left_vertices_per_label = np.append(left_vertices_per_label, label.vertices)
            # each vertex will show the average correlation for the corresponding region
            maxes_to_add = np.ones(len(label.vertices)) * max_over_time_left[region_index_in_results]
            left_maxes_per_label = np.append(left_maxes_per_label,
                                                    maxes_to_add)
        # else:
        #     print(max_over_time_left[region_index_in_results])
    # the vertices are required to be sorted for the plotting function
    left_sorted_vertices_array = sorted(left_vertices_per_label)
    left_sorted_vertices_array_inds = np.argsort(left_vertices_per_label)

    for label in label_right:
        label_name = label.name[:-3]
        # print(label_name)
        if label_name not in REGIONS:
            continue
        region_index_in_results = REGIONS.index(label_name)
        if max_over_time_right[region_index_in_results] >= acc_thresh:  # include this regiion in the plotting

            right_vertices_per_label = np.append(right_vertices_per_label, label.vertices)
            # each vertex will show the average correlation for the corresponding region
            maxes_to_add = np.ones(len(label.vertices)) * max_over_time_right[region_index_in_results]
            # if label_name == 'frontalpole':
            #     maxes_to_add[0:200] = 1.0
            right_maxes_per_label = np.append(right_maxes_per_label,
                                                     maxes_to_add)
    # the vertices are required to be sorted for the plotting function
    right_sorted_vertices_array = sorted(right_vertices_per_label)
    right_sorted_vertices_array_inds = np.argsort(right_vertices_per_label)

    # so sort the correlations the same way
    maxes_in_order = np.transpose([np.append(left_maxes_per_label[left_sorted_vertices_array_inds],
                                            right_maxes_per_label[right_sorted_vertices_array_inds])])
    print(np.max(maxes_in_order))
    # maxes_in_order *= 100.0


    left_sorted_vertices_array = [int(ind) for ind in left_sorted_vertices_array]
    right_sorted_vertices_array = [int(ind) for ind in right_sorted_vertices_array]

    # create source estimate object with data to show for the left and right hemispheres, and corresponding locations
    src = mne.SourceEstimate(maxes_in_order, [left_sorted_vertices_array, right_sorted_vertices_array], tmin=0,
                             tstep=1)


    cmap = 'jet'

    lims = [0.0, 0.5, 1.0]  # based on min and max avrg correlation values over all models
    smoothing_steps = 1
    bk = 'white'
    fname = "/home/nrafidi/thesis_figs/PassAct3_pooled_{}_eos-{}_{}_acc{}_{}".format(word, args.plot_type, time_win_str, acc_thresh_str, args.surface)

    # f0 = mne.viz.plot_source_estimates(src, subject=STRUCTURAL, background=bk, surface='inflated', hemi='lh', views='lat',
    #                                   clim={'kind': 'value', 'lims': lims}, colormap=cmap, subjects_dir=SUBJ_DIR,
    #                                   smoothing_steps=smoothing_steps, spacing='ico4', backend='matplotlib')
    #
    # f0.savefig(fname + "_lh_med_source_plot.pdf", bbox_inches='tight')

    f1 = plt.figure(figsize=(15, 15))
    f1 = mne.viz.plot_source_estimates(src, figure=f1, subject=STRUCTURAL, background=bk, surface=args.surface, hemi='lh', views='med',
                                      clim={'kind': 'value', 'lims': lims}, colormap=cmap, subjects_dir=SUBJ_DIR,
                                      smoothing_steps=smoothing_steps, spacing='ico4', backend='matplotlib')
    f1.savefig(fname + "_lh_lat_source_plot.pdf", bbox_inches='tight')

    f2 = plt.figure(figsize=(15, 15))
    f2 = mne.viz.plot_source_estimates(src, figure=f2, subject=STRUCTURAL, background=bk, surface=args.surface, hemi='rh', views='lat',
                                      clim={'kind': 'value', 'lims': lims}, colormap=cmap, subjects_dir=SUBJ_DIR,
                                      smoothing_steps=smoothing_steps, spacing='ico4', backend='matplotlib')
    f2.savefig(fname + "_rh_lat_source_plot.pdf", bbox_inches='tight')

    # f3 = mne.viz.plot_source_estimates(src, subject=STRUCTURAL, background=bk, surface='inflated', hemi='rh', views='med',
    #                                   clim={'kind': 'value', 'lims': lims}, colormap=cmap, subjects_dir=SUBJ_DIR,
    #                                   smoothing_steps=smoothing_steps, spacing='ico4', backend='matplotlib')
    #
    #
    # f3.savefig(fname + "_rh_med_source_plot.pdf", bbox_inches='tight')

    # fig, ax = plt.subplots()
    # norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    #
    # cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                        norm=norm,
    #                                        orientation='vertical')
    #
    # ax.tick_params(labelsize=20)
    # plt.show()
