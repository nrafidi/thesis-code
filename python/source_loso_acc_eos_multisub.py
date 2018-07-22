import argparse
import load_data_ordered as load_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

PLOT_TITLE_EXP = {'krns2': 'Pilot Experiment',
                  'PassAct3': 'Final Experiment'}
PLOT_TITLE_SEN = {'active': 'Active Sentences',
                  'passive': 'Passive Sentences',
                  'pooled': 'All Sentences'}

PLOT_TITLE_WORD = {'noun1': 'First Noun',
                  'verb': 'Verb',
                   'agent': 'Agent',
                   'patient': 'Patient',
                   'voice': 'Sentence Voice',
                   'propid': 'Proposition ID',
                   'senlen': 'Sentence Length',
                   'bind': 'Argument Binding'}

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
    args = parser.parse_args()

    ticklabelsize = 14
    legendfontsize = 16
    axislabelsize = 18
    suptitlesize = 25
    axistitlesize = 20
    axislettersize = 20

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

    time_step = int(250 / args.overlap)
    time_adjust = args.win_len * 0.002 * time_step

    source_by_time_mat = []

    for hemi in HEMIS:
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
            mean_acc = np.mean(multi_fold_acc, axis=0)
            diag_acc = np.diag(mean_acc)
            source_by_time_mat.append(diag_acc[None, ...])


    source_by_time_mat = np.concatenate(source_by_time_mat)

    combo_fig, ax = plt.subplots(figsize=(12, 18))
    im = ax.imshow(source_by_time_mat, interpolation='nearest', aspect='auto',
                   vmin=0.0, vmax=1.0)

    combo_fig.suptitle('Rank Accuracy for {word} Decoding'.format(
        word=PLOT_TITLE_WORD[word]), fontsize=suptitlesize)

    num_time = len(time_win)
    ax.set_xticks(np.arange(0, num_time, time_step))

    min_time = 0.0
    max_time = 0.5 * len(time_win) / time_step
    label_time = np.arange(min_time, max_time, 0.5) + args.win_len*0.002
    ax.set_xticklabels(label_time)

    ax.set_yticks(np.arange(2*len(REGIONS)))
    ax.set_yticklabels(REGIONS + REGIONS)
    ax.tick_params(labelsize=ticklabelsize)

    ax.axvline(x=0.625*time_step, color='k')
    ax.axhline(y=len(REGIONS) - 0.5, color='k')
    ax.text(-0.25, 0.75, 'Left', transform=ax.transAxes,
            rotation=90, rotation_mode='anchor', size=axislettersize)
    ax.text(-0.25, 0.25, 'Right', transform=ax.transAxes,
            rotation=90, rotation_mode='anchor', size=axislettersize)

    cbar = combo_fig.colorbar(im)

    combo_fig.text(-0.25, 0.5, 'ROI', va='center',
                   rotation=90, rotation_mode='anchor', fontsize=axislabelsize, weight='bold')
    combo_fig.text(0.5, 0.04, 'Time Relative to Last Word Onset (s)', ha='center', fontsize=axislabelsize)
    combo_fig.savefig(
            '/home/nrafidi/thesis_figs/{exp}_eos_avg-source_multisub_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.pdf'.format(
                exp=experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime, avgTest=args.avgTest,
                win_len=args.win_len,
                overlap=args.overlap,
                num_instances=args.num_instances
            ), bbox_inches='tight')
    combo_fig.savefig(
        '/home/nrafidi/thesis_figs/{exp}_eos_avg-source_multisub_{sen_type}_{word}_{alg}_win{win_len}_ov{overlap}_ni{num_instances}_avgTime{avgTime}_avgTest{avgTest}.png'.format(
            exp=experiment, sen_type=sen_type, word=word, alg=args.alg, avgTime=args.avgTime,
            avgTest=args.avgTest,
            win_len=args.win_len,
            overlap=args.overlap,
            num_instances=args.num_instances
        ), bbox_inches='tight')
    plt.show()