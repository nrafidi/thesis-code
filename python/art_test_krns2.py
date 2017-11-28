import argparse
import itertools
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import run_SV
import scipy.io as sio
import os.path


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

NUM_FOLDSS = [16, 32, 160]
ADJS = [None, 'mean_center', 'zscore']
NUM_INSTANCESS = [1, 2, 10]  # 5 10


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='B')
    args = parser.parse_args()

    exp = 'krns2'
    sub = args.subject

    top_dir = run_SV.TOP_DIR.format(exp=exp)
    save_dir = run_SV.SAVE_DIR.format(top_dir=top_dir, sub=sub)

    param_grid = itertools.product(NUM_FOLDSS,
                                   ADJS,
                                   NUM_INSTANCESS)

    sen_type = args.sen_type
    word = args.word
    art1_str = args.art1_str
    art2_str = args.art2_str



    scores = []
    score_maxes = []
    grid_list = []
    for grid in param_grid:
        fname = run_SV.SAVE_FILE.format(dir=save_dir,
                                 sub='B',
                                 sen_type=sen_type,
                                 word=word,
                                 model='one_hot',
                                 art1=art1_str,
                                 art2=art2_str,
                                 direction='encoding',
                                 pca='F',
                                 pdtw='F',
                                 perm='F',
                                 num_folds=grid[0],
                                 alg='ols',
                                 adj=grid[1],
                                 inst=grid[2],
                                 rep=10,
                                 rsP=1,
                                 rsC=run_SV.CV_RAND_STATE,
                                 rsS=run_SV.SUB_CV_RAND_STATE)
        if os.path.isfile(fname + '.npz'):
            result = np.load(fname + '.npz')
            curr_score = result['scores']
            scores.append(curr_score)
            score_maxes.append(np.max(curr_score))
            grid_list.append(grid)

    time = result['time']
    time[np.abs(time) <= 1e-14] = 0.0
    num_time = time.size
    fig0, ax0 = plt.subplots()
    ax0.hist(np.array(score_maxes))
    i_max = np.argmax(score_maxes)
    # i_max = grid_list.index((160, 'zscore', 10))
    print('Best score for params {} was {}'.format(grid_list[i_max], score_maxes[i_max]))

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    best_score = np.reshape(scores[i_max], (306, -1))
    best_score = best_score[sorted_inds, :]

    fig, ax = plt.subplots()
    h = ax.imshow(best_score, interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0)
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, num_time, 250))
    ax.set_xticklabels(time[::250])
    ax.set_xlabel('Time')
    ax.set_title('{} {} {} {}\nArt1: {} Art2: {}'.format(args.experiment, args.subject, args.sen_type, args.word,
                                                         art1_str, art2_str))
    plt.colorbar(h)
    plt.savefig('POVE_{}_{}_{}_{}_art1{}_art2{}.pdf'.format(args.experiment, args.subject, args.sen_type, args.word,
                                                            art1_str, art2_str), bbox_inches='tight')
    plt.show()
