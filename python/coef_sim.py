import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as matlib
import agg_TGM


def coef_by_param(exp,
                  word,
                  sen_type,
                  accuracy,
                  sub,
                  param_specs):
    overlap = param_specs['o']
    sub_results, sub_params, sub_time = agg_TGM.agg_results(exp,
                                                            'coef',
                                                            word,
                                                            sen_type,
                                                            accuracy,
                                                            sub,
                                                            param_specs=param_specs)

    sim_mat = np.empty((len(sub_results), len(sub_results[0]) - 1))
    min_win = 1000
    for i_win, res in enumerate(sub_results):
        win_size = int(sub_params['w'][i_win])
        if len(res) < min_win:
            min_win = len(res)
        for i_coef in range(1, len(res)):
            prev_res = np.zeros(res[i_coef - 1].shape)
            prev_res[res[i_coef - 1] < 0] = -1.0
            prev_res[res[i_coef - 1] > 0] = 1.0

            curr_res = np.zeros(res[i_coef].shape)
            curr_res[res[i_coef] < 0] = -1.0
            curr_res[res[i_coef] > 0] = 1.0

            prev_mat = np.reshape(np.sum(prev_res, axis=0), (win_size, 306))
            curr_mat = np.reshape(np.sum(curr_res, axis=0), (win_size, 306))

            if overlap < prev_mat.shape[0]:
                np.roll(prev_mat, -overlap)
                end_ind = prev_mat.shape[0] - overlap
                prev_mat = prev_mat[:end_ind, :]
                curr_mat = curr_mat[:end_ind, :]

            #         if win_size > overlap:
            #             f, axs = plt.subplots(2, 1)ddd
            #             axs[0].imshow(prev_mat, interpolation='nearest', aspect='auto')
            #             axs[1].imshow(curr_mat, interpolation='nearest', aspect='auto')
            #              plt.show()
            sim_mat[i_win, i_coef - 1] = np.linalg.norm(curr_mat - prev_mat)
    sim_mat = sim_mat[:, 0:(min_win - 1)]
    print(sim_mat[0, 0])
    meow = np.max(sim_mat, axis=1)
    print(meow)
    sim_mat = np.divide(sim_mat, meow[:, None])
    woof = np.min(sim_mat, axis=1)
    sim_mat -= woof[:, None]
    print(sim_mat[0, 0])
    return sim_mat

if __name__ == '__main__':
    exp = 'krns2'
    mode = 'coef'
    word = 'firstNoun'
    sen_type = 'active'
    accuracy = 'abs'
    sub = 'B'
    overlap = 12

    param_specs = {'o': overlap,
                   'pd': 'F',
                   'pr': 'F',
                   'alg': 'LR',
                   'F': 2,
                   'z': 'F',
                   'avg': 'F',
                   'ni': 2,
                   'nr': 10,
                   'rs': 1}

    sim_mat = coef_by_param(exp,
                            word,
                            sen_type,
                            accuracy,
                            sub,
                            param_specs)

    f, ax = plt.subplots()
    ax.imshow(sim_mat, interpolation='nearest', aspect='auto')
#    ax.plot(sim_mat[4, 0:100])
    plt.show()
