import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import string
import run_coef_TGM_multisub
from mne.viz import plot_topomap
from syntax_vs_semantics import sensor_metadata


TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
MULTI_SAVE_FILE = '{dir}TGM-LOSO_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{rank_str}{mode}'

SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'


PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                    'noun2': 'Second Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice'}


def _make_coordinates(sensor_layout, sensor_type_id):
    coordinates = np.hstack(
        [np.expand_dims(np.array([si.x_coord for si in sensor_layout if si.sensor_type_id == sensor_type_id]), 1),
         np.expand_dims(np.array([si.y_coord for si in sensor_layout if si.sensor_type_id == sensor_type_id]), 1)])
    # coordinates is now sensors by xy
    return coordinates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--win_len', type=int, default=50)
    parser.add_argument('--overlap', type=int, default=5)
    parser.add_argument('--alg', default='lr-l2', choices=['lr-l2', 'lr-l1'])
    parser.add_argument('--adj', default='zscore', choices=['None', 'mean_center', 'zscore'])
    parser.add_argument('--num_instances', type=int, default=2)
    parser.add_argument('--avgTime', default='T')
    parser.add_argument('--avgTest', default='T')
    parser.add_argument('--sensor_type_id', type=int, default=2)
    args = parser.parse_args()

    if args.avgTime == 'T':
        aT = 'Time Average '
    else:
        aT = ''
    if args.avgTest == 'T':
        aTst = 'Test Average'
    else:
        aTst = ''

    if args.avgTime == 'T':
        avg_time_str = 'Time Average'
    else:
        avg_time_str = 'No Time Average'

    if args.avgTest == 'T':
        avg_test_str = 'Test Sample Average'
    else:
        avg_test_str = 'No Test Sample Average'

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

    top_dir = TOP_DIR.format(exp=args.experiment)

    sensor_layout = sensor_metadata.read_sensor_metadata()
    coordinates = _make_coordinates(sensor_layout, args.sensor_type_id)
    sensors_to_keep = [i_s for i_s, s in enumerate(sensor_layout) if s.sensor_type_id == args.sensor_type_id]

    word_list = ['noun1', 'verb', 'noun2']
    num_plots = len(word_list)
    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step
    combo_fig = plt.figure(figsize=(6*num_plots, 12))
    combo_grid = AxesGrid(combo_fig, 111, nrows_ncols=(2, num_plots),
                          axes_pad=0.7, cbar_mode='single', cbar_location='right',
                          cbar_pad=0.5, share_all=True)
    i_combo = 0
    for i_sen, sen_type in enumerate(['active', 'passive']):
        for i_word, word in enumerate(word_list):

            fname = run_coef_TGM_multisub.SAVE_FILE.format(dir=top_dir,
                                                           sen_type=sen_type,
                                                           word=word,
                                                           win_len=args.win_len,
                                                           ov=args.overlap,
                                                           alg=args.alg,
                                                           adj=args.adj,
                                                           avgTm=args.avgTime,
                                                           inst=args.num_instances)

            result = np.load(fname + '.npz')
            maps = result['haufe_maps']
            win_starts = result['win_starts']
            time_win = result['time'][win_starts]

            rank_file = MULTI_SAVE_FILE.format(dir=top_dir,
                                                sen_type=sen_type,
                                                word=word,
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

            rank_result = np.load(rank_file + '.npz')
            acc_all = rank_result['tgm_rank']
            mean_acc = np.mean(acc_all, axis=0)


            max_acc = np.argmax(np.diag(mean_acc))
            time_to_plot = time_win[max_acc]

            map = maps[max_acc]
            class_map = np.mean(map, axis=0)
            sub_map = np.zeros((306,))
            for i_sub in range(len(run_coef_TGM_multisub.VALID_SUBS[args.experiment])):
                start_ind = i_sub * 306
                end_ind = start_ind + 306
                sub_map += class_map[start_ind:end_ind]
            sub_map /= len(run_coef_TGM_multisub.VALID_SUBS[args.experiment])
            sub_map = sub_map[sensors_to_keep]

            print(np.min(sub_map))
            print(np.max(sub_map))

            ax = combo_grid[i_combo]
            print(i_combo)
            im, _ = plot_topomap(sub_map, coordinates, axes=ax, show=False)
            ax.set_title('%s: %.2f-%.2f' % (PLOT_TITLE_WORD[word], time_to_plot, time_to_plot + args.win_len * 0.002), fontsize=axistitlesize)

            ax.text(-0.12, 1.02, string.ascii_uppercase[i_combo], transform=ax.transAxes,
                                    size=axislettersize, weight='bold')
            i_combo += 1
        


    cbar = combo_grid.cbar_axes[0].colorbar(im)
    combo_fig.suptitle('Classifier Importance Maps at Peak Accuracy Time',
        fontsize=suptitlesize)
    combo_fig.text(0.488, 0.8575, 'Active', ha='center', fontsize=axistitlesize+2)
    combo_fig.text(0.488, 0.475, 'Passive', ha='center', fontsize=axistitlesize+2)
    combo_fig.subplots_adjust(top=0.85)
    combo_fig.savefig('/home/nrafidi/thesis_figs/{exp}_helmet{id}_multisub_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                    exp=args.experiment, id=args.sensor_type_id, sen_type='both', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                    win_len=args.win_len,
                    overlap=args.overlap,
                    num_instances=args.num_instances
                ), bbox_inches='tight')

    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_helmet{id}_{sen_type}_multisub_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=args.experiment, id=args.sensor_type_id, sen_type='both', word='all', alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')

    plt.show()