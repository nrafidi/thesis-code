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
    num_folds = 16
    adj = None
    num_instances = 1


    top_dir = run_SV.TOP_DIR.format(exp=exp)
    save_dir = run_SV.SAVE_DIR.format(top_dir=top_dir, sub=sub)

    # Active Second Noun First Article
    sen_type = 'active'
    word = 'secondNoun'
    art1_str = 'T'
    art2_str = 'T'

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
                             num_folds=num_folds,
                             alg='ols',
                             adj=adj,
                             inst=num_instances,
                             rep=10,
                             rsP=1,
                             rsC=run_SV.CV_RAND_STATE,
                             rsS=run_SV.SUB_CV_RAND_STATE)
    result = np.load(fname + '.npz')
    all_feat_res = result['scores']

    art1_str = 'O'
    art2_str = 'O'

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
                                    num_folds=num_folds,
                                    alg='ols',
                                    adj=adj,
                                    inst=num_instances,
                                    rep=10,
                                    rsP=1,
                                    rsC=run_SV.CV_RAND_STATE,
                                    rsS=run_SV.SUB_CV_RAND_STATE)
    result = np.load(fname + '.npz')
    only_art_res = result['scores']


    time = result['time']
    time[np.abs(time) <= 1e-10] = 0.0
    num_time = time.size

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    score_diff = np.reshape(all_feat_res - only_art_res, (306, -1))
    score_diff = score_diff[sorted_inds, :]
    print(np.max(score_diff))
    fig, ax = plt.subplots()
    h = ax.imshow(score_diff, interpolation='nearest', aspect='auto', vmin=0.0, vmax=0.4)
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, num_time, 250))
    ax.set_xticklabels(time[::250])
    ax.set_xlabel('Time')
    plt.colorbar(h)
    ax.set_title('Active\nSecondNoun + Art - Art')
    plt.savefig('POVE_diff_active_secondNoun.pdf', bbox_inches='tight')

    # Passive First Noun Second Article
    sen_type = 'passive'
    word = 'firstNoun'
    art1_str = 'T'
    art2_str = 'T'

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
                                    num_folds=num_folds,
                                    alg='ols',
                                    adj=adj,
                                    inst=num_instances,
                                    rep=10,
                                    rsP=1,
                                    rsC=run_SV.CV_RAND_STATE,
                                    rsS=run_SV.SUB_CV_RAND_STATE)
    result = np.load(fname + '.npz')
    all_feat_res = result['scores']

    art1_str = 'O'
    art2_str = 'O'

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
                                    num_folds=num_folds,
                                    alg='ols',
                                    adj=adj,
                                    inst=num_instances,
                                    rep=10,
                                    rsP=1,
                                    rsC=run_SV.CV_RAND_STATE,
                                    rsS=run_SV.SUB_CV_RAND_STATE)
    result = np.load(fname + '.npz')
    only_art_res = result['scores']

    time = result['time']
    time[np.abs(time) <= 1e-10] = 0.0
    num_time = time.size

    sorted_inds, sorted_reg = sort_sensors()
    uni_reg = np.unique(sorted_reg)
    yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]

    score_diff = np.reshape(all_feat_res - only_art_res, (306, -1))
    score_diff = score_diff[sorted_inds, :]
    print(np.max(score_diff))
    fig, ax = plt.subplots()
    h = ax.imshow(score_diff, interpolation='nearest', aspect='auto', vmin=0.0, vmax=0.4)
    ax.set_yticks(yticks_sens)
    ax.set_yticklabels(uni_reg)
    ax.set_ylabel('Sensors')
    ax.set_xticks(range(0, num_time, 250))
    ax.set_xticklabels(time[::250])
    ax.set_xlabel('Time')
    plt.colorbar(h)
    ax.set_title('Passive\nFirstNoun + Art - Art')
    plt.savefig('POVE_diff_passive_firstNoun.pdf', bbox_inches='tight')


    plt.show()
