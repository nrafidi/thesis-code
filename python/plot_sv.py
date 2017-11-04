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
    parser.add_argument('--experiment', default='krns2')
    parser.add_argument('--subject', default='B')
    parser.add_argument('--sen_type', default='active')
    parser.add_argument('--word', default='firstNoun')
    args = parser.parse_args()

    top_dir = run_SV.TOP_DIR.format(exp=args.experiment)
    save_dir = run_SV.SAVE_DIR.format(top_dir=top_dir, sub=args.subject)

    param_grid = itertools.product(NUM_FOLDSS,
                                   ADJS,
                                   NUM_INSTANCESS)

    scores = []
    for grid in param_grid:
        fname = run_SV.SAVE_FILE.format(dir=save_dir,
                                        sub=args.subject,
                                        sen_type=args.sen_type,
                                        word=args.word,
                                        pdtw='F',
                                        perm='F',
                                        num_folds=grid[0],
                                        alg='ridge',
                                        adj=grid[1],
                                        inst=grid[2],
                                        rep=10,
                                        rsP=1,
                                        rsC=run_SV.CV_RAND_STATE,
                                        rsS=run_SV.SUB_CV_RAND_STATE)
        if os.path.isfile(fname + '.npz'):
            result = np.load(fname + '.npz')
            curr_score = result['scores']
            scores.append(np.max(curr_score))

    time = result['time']
    time[time <= 1e-14] = 0.0
    num_time = time.size

    i_max = np.argmax(scores)
    print('Best score was {} for params {}'.format(scores[i_max], param_grid[i_max]))

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    best_score = np.reshape(scores[i_max], (306, -1))
    best_score = best_score[sorted_inds, :]

    fig, ax = plt.subplots()
    h = ax.imshow(best_score, interpolation='nearest', aspect='auto')
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, num_time, 250))
    ax.set_xticklabels(time[::250])
    ax.set_xlabel('Time')
    plt.colorbar(h)
    plt.show()
