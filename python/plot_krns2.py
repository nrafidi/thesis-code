import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
import numpy as np
import load_data
from scipy.stats.mstats import zscore
import agg_TGM
import run_TGM
import coef_sim


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'pred'
    # word = 'firstNoun'
    # sen_type = 'active'
    accuracy = 'abs'

    for word in ['firstNoun', 'verb', 'secondNoun']:
        for sen_type in ['passive', 'active']:
            for o in [6]:
                param_specs = {'o': o,
                               'pd': 'F',
                               'pr': 'F',
                               'alg': 'GNB',
                               'F': 2,
                               'z': 'F',
                               'avg': 'F',
                               'ni': 2,
                               'nr': 10,
                               'rsPerm': 1,
                               'rsCV': run_TGM.CV_RAND_STATE,
                               'rsSCV': run_TGM.SUB_CV_RAND_STATE}
                sub_results = {}
                sub_params = {}
                sub_time = {}
                sub_sim = []
                for sub in ['B']: #load_data.VALID_SUBS[exp]:

                    sub_results[sub], sub_params[sub], sub_time[sub] = agg_TGM.agg_results(exp,
                                                                                           mode,
                                                                                           word,
                                                                                           sen_type,
                                                                                           accuracy,
                                                                                           sub,
                                                                                           param_specs=param_specs)
                    # sub_sim.append(coef_sim.coef_by_param(exp,
                    #                                       word,
                    #                                       sen_type,
                    #                                       accuracy,
                    #                                       sub,
                    #                                       param_specs, param_limit=200))
                diag, param_val, time, _ = agg_TGM.get_diag_by_param(sub_results, sub_params, sub_time, 'w', {})
                diag = np.mean(diag, axis=0)
                for i_win in range(diag.shape[0]):
                    fig, ax = plt.subplots()
                    ax.plot(diag[i_win, :])
                plt.show()


                # sim_mat = np.mean(np.asarray(sub_sim), axis=0)
                #
                # sim_mat -= np.min(sim_mat, axis=1)[:, None]
                # sim_mat = np.divide(sim_mat, np.max(sim_mat, axis=1)[:, None])
                # num_win = diag.shape[0]
                #
                # diag_max = np.array([np.max(diag[i_win, :]) for i_win in range(num_win)])
                # win_sizes = np.array([param_val[0, i_win]*2 for i_win in range(num_win)])
                # # print(diag_max.shape)
                # # print(win_sizes.shape)
                # fig, ax = plt.subplots()
                # ax.plot(win_sizes, diag_max)
                # # plt.show()
                #
                # new_diag = np.empty(diag.shape)
                # plot_time = np.squeeze(time[0, :, :])
                # fig, axs = plt.subplots(num_win, 1, figsize=(20, 30))
                # for i_win in range(num_win):
                #     win_len = param_val[0, i_win]*0.002
                #     time_to_plot = plot_time[i_win, :] + win_len*0.5
                #     diag_to_plot = diag[i_win, :]
                #     axs[i_win].plot(time_to_plot, diag_to_plot)
                #     sim_to_plot = sim_mat[i_win, :]
                #     # sim_to_plot[sim_to_plot > 0] = 0.6
                #     # sim_to_plot[sim_to_plot < 0] = 0.1
                #     axs[i_win].plot(time_to_plot[:sim_to_plot.shape[0]], sim_to_plot)
                #     axs[i_win].plot(time_to_plot, np.ones(time_to_plot.shape)*0.25, '--')
                #     diag_to_plot[diag_to_plot >= 0.3] = 0.75
                #     diag_to_plot[diag_to_plot < 0.3] = 0
                #     axs[i_win].plot(time_to_plot, diag_to_plot, ':')
                #     new_diag[i_win, :] = diag_to_plot
                #     axs[i_win].set_ylim([0.1, 1])
                #     axs[i_win].set_xlim([np.min(plot_time), np.max(plot_time)])
                #     axs[i_win].set_title(win_len)
                #
                # fig.suptitle('{} {}\nOverlap {}'.format(word, sen_type, o), fontsize=16)
                # plt.savefig('overlap{}_{}_{}_sim_acc.pdf'.format(o, word, sen_type), bbox_inches='tight')
                # fig, ax = plt.subplots()
                # ax.plot(np.sum(np.divide(new_diag, 0.75), axis=0))
                # # ax.plot(new_diag[-1, :] + 2)
                # # ax.plot(new_diag[0, :] + 1)
                # ax.set_yticks(range(1, num_win+1))
                # ax.set_ylabel('Number of windows')
                # ax.set_ylim([0, num_win])
                # ax.set_xticks(range(0, time.shape[-1], 25))
                # ax.set_xticklabels(np.squeeze(time[0, -1, ::25]))
                # ax.set_title('Number of windows above chance over time\n{} {} Overlap {}'.format(word, sen_type, o))
                # plt.savefig('overlap{}_{}_{}_comb_acc.pdf'.format(o, word, sen_type), bbox_inches='tight')
                # plt.show()


            #     thing_to_plot = 1
            #     fig, ax = plt.subplots()
            #     ax.plot(diag[thing_to_plot, :])
            #  #   im = ax.imshow(diag, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
            #  #   ax.set_yticks(range(param_val.shape[-1]))
            # #    ax.set_yticklabels(param_val[0, :].astype('int'))
            #     ax.set_xticks(range(0, time.shape[-1], 5))
            #     print(time.shape)
           #     ax.set_xticklabels(np.squeeze(time[0, thing_to_plot, ::5]))
        #   plt.colorbar(im)
         #       plt.savefig('plot.pdf', bbox_inches='tight')
         #         plt.plot(diag[0, :])
         #        plt.show()
