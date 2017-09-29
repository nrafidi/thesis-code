import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import load_data
from scipy.stats.mstats import zscore
import agg_TGM
import coef_sim


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'pred'
    word = 'firstNoun'
    sen_type = 'active'
    accuracy = 'abs'

    for o in [12, 25]:
        param_specs = {'o': o,
                       'pd': 'F',
                       'pr': 'F',
                       'alg': 'LR',
                       'F': 2,
                       'z': 'F',
                       'avg': 'F',
                       'ni': 2,
                       'nr': 10,
                       'rs': 1}
        sub_results = {}
        sub_params = {}
        sub_time = {}
        for sub in ['B']:  # load_data.VALID_SUBS[exp]:

            sub_results[sub], sub_params[sub], sub_time[sub] = agg_TGM.agg_results(exp,
                                                                                   mode,
                                                                                   word,
                                                                                   sen_type,
                                                                                   accuracy,
                                                                                   sub,
                                                                                   param_specs=param_specs)
        diag, param_val, time, _ = agg_TGM.get_diag_by_param(sub_results, sub_params, sub_time, 'w', {})
        diag = np.mean(diag, axis=0)

    #     thing_to_plot = 1
    #     fig, ax = plt.subplots()
    #     ax.plot(diag[thing_to_plot, :])
    #  #   im = ax.imshow(diag, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    #  #   ax.set_yticks(range(param_val.shape[-1]))
    # #    ax.set_yticklabels(param_val[0, :].astype('int'))
    #     ax.set_xticks(range(0, time.shape[-1], 5))
    #     print(time.shape)
    #     ax.set_xticklabels(np.squeeze(time[0, thing_to_plot, ::5]))
    #  #   plt.colorbar(im)
    #     #plt.savefig('plot.pdf', bbox_inches='tight')
    #     # plt.plot(diag[0, :])
    #     plt.show()

        sim_mat = coef_sim.coef_by_param(exp,
                                         word,
                                         sen_type,
                                         accuracy,
                                         'B',
                                         param_specs)

        sim_mat = zscore(sim_mat, axis=1)
        num_win = sim_mat.shape[0]

        fig, axs = plt.subplots(num_win, 1, figsize=(20, 30))
        for i_win in range(num_win):
            diag_to_plot = diag[i_win, :]
            diag_to_plot[diag_to_plot > 0.25] = 0.5
            diag_to_plot[diag_to_plot < 0.25] = 0
            axs[i_win].plot(diag_to_plot)
            sim_to_plot = sim_mat[i_win, :]
            sim_to_plot[sim_to_plot > 0] = 0.5
            sim_to_plot[sim_to_plot < 0] = 0
            axs[i_win].plot(sim_to_plot)
            axs[i_win].set_ylim([0, 0.75])
        fig.suptitle('Overlap {}'.format(o))
        plt.savefig('overlap{}.pdf'.format(o))



