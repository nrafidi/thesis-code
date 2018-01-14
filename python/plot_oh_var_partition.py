import argparse
import itertools
import matplotlib
matplotlib.use('TkAgg')
import load_data_ordered as load_data
import matplotlib.pyplot as plt
import numpy as np
import run_OH_Reg
import scipy.io as sio
import os.path


SENSOR_MAP = '/bigbrain/bigbrain.usr1/homes/nrafidi/MATLAB/groupRepo/shared/megVis/sensormap.mat'

NUM_FEATS = {'krns2': 16,
             'PassAct2': 12,
             'PassAct3': 12}


def sort_sensors():
    load_var = sio.loadmat(SENSOR_MAP)
    sensor_reg = load_var['sensor_reg']
    sensor_reg = [str(sens[0][0]) for sens in sensor_reg]
    sorted_inds = np.argsort(sensor_reg)
    sorted_reg = [sensor_reg[ind] for ind in sorted_inds]
    return sorted_inds, sorted_reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['krns2', 'PassAct2', 'PassAct3'], default='krns2')
    parser.add_argument('--sen_type', choices=['active', 'passive'], default='active')
    parser.add_argument('--word', choices=['noun1', 'verb', 'noun2'], default='noun1')
    parser.add_argument('--adj', choices=[None, 'mean_center', 'zscore'], default=None)
    args = parser.parse_args()

    exp = args.experiment
    subjects = load_data.VALID_SUBS[exp]
    num_feats_all = NUM_FEATS[exp]
    sen_type = args.sen_type
    word = args.word
    adj = args.adj

    top_dir = run_OH_Reg.TOP_DIR.format(exp=exp)


    total_est_all = []
    total_true_all = []
    total_est_word = []
    total_true_word = []
    for subject in subjects:
        save_dir = run_OH_Reg.SAVE_DIR.format(top_dir=top_dir, sub=subject)
        fname = run_OH_Reg.SAVE_FILE.format(dir=save_dir,
                                 sub=subject,
                                 sen_type=sen_type,
                                 word='all',
                                 perm='F',
                                 num_folds=16,
                                 alg='ols',
                                 adj=adj,
                                 inst=1,
                                 rep=10,
                                 rsP=1,
                                 rsC=run_OH_Reg.CV_RAND_STATE)
        if os.path.isfile(fname + '.npz'):
            result = np.load(fname + '.npz')
            total_est_all.append(result['preds'])
            total_true_all.append(result['test_data_all'])

        fname = run_OH_Reg.SAVE_FILE.format(dir=save_dir,
                                            sub=subject,
                                            sen_type=sen_type,
                                            word=word,
                                            perm='F',
                                            num_folds=16,
                                            alg='ols',
                                            adj=adj,
                                            inst=1,
                                            rep=10,
                                            rsP=1,
                                            rsC=run_OH_Reg.CV_RAND_STATE)
        if os.path.isfile(fname + '.npz'):
            result = np.load(fname + '.npz')
            total_est_word.append(result['preds'])
            total_true_word.append(result['test_data_all'])

    total_est_all = np.concatenate(total_est_all, axis=0)
    total_true_all = np.concatenate(total_true_all, axis=0)
    total_est_word = np.concatenate(total_est_word, axis=0)
    total_true_word = np.concatenate(total_true_word, axis=0)

    print(total_est_all.shape)

    # time = result['time']
    # time[np.abs(time) <= 1e-14] = 0.0
    # num_time = time.size
    # fig0, ax0 = plt.subplots()
    # ax0.hist(np.array(score_maxes))
    # # i_max = np.argmax(score_maxes)
    # i_max = grid_list.index((16, None, 1))
    # print('Best score for params {} was {}'.format(grid_list[i_max], score_maxes[i_max]))
    #
    # sorted_inds, sorted_reg = sort_sensors()
    # uni_reg = np.unique(sorted_reg)
    # yticks_sens = [sorted_reg.index(reg) for reg in uni_reg]
    #
    # best_score = np.reshape(scores[i_max], (306, -1))
    # best_score = best_score[sorted_inds, :]
    #
    # fig, ax = plt.subplots()
    # h = ax.imshow(best_score, interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0)
    # ax.set_yticks(yticks_sens)
    # ax.set_yticklabels(uni_reg)
    # ax.set_ylabel('Sensors')
    # ax.set_xticks(range(0, num_time, 250))
    # ax.set_xticklabels(time[::250])
    # ax.set_xlabel('Time')
    # ax.set_title('{} {} {} {}\nArt1: {} Art2: {}'.format(args.experiment, args.subject, args.sen_type, args.word,
    #                                                      art1_str, art2_str))
    # plt.colorbar(h)
    # plt.savefig('POVE_{}_{}_{}_{}_art1{}_art2{}.pdf'.format(args.experiment, args.subject, args.sen_type, args.word,
    #                                                         art1_str, art2_str), bbox_inches='tight')
    # plt.show()
