import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import agg_TGM

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
    sub_results, sub_params, sub_time = agg_results(exp,
                                                    mode,
                                                    word,
                                                    sen_type,
                                                    accuracy,
                                                    sub,
                                                    param_specs=param_specs)

    sim_mat = np.empty((len(sub_results), len(sub_results[0]) - 1))
    for i_win, res in enumerate(sub_results):
        win_size = int(sub_params['w'][i_win])

        for i_coef in range(1, len(res)):
            prev_mat = np.reshape(np.sum(res[i_coef-1], axis=0), (win_size, 306))
            curr_mat = np.reshape(np.sum(res[i_coef], axis=0), (win_size, 306))
            prev_mat = np.roll(prev_mat, overlap)

            end_ind = prev_mat.shape[0] - overlap
            prev_mat = prev_mat[:end_ind, :]
            curr_mat = curr_mat[:end_ind, :]

            f, axs = plt.subplots(1, 2, sharex=True)
            axs[0].imshow(prev_mat, interpolation='nearest')
            axs[1].imshow(curr_mat, interpolation='nearest')
            plt.show()
            sim_mat[i_win, i_coef-1] = np.linalg.norm(curr_mat - prev_mat)

    f, ax = plt.subplot()
    ax.plot(sim_mat[0, :])
    plt.show()