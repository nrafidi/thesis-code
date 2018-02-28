import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import load_data_ordered as load_data_mine
from syntax_vs_semantics import load_data as load_data_new


if __name__ == '__main__':
    tmin = -0.5
    tmax = 4.0
    data_old, labels_old, time_old, final_inds_old = load_data_mine.load_PassAct3_matlab(subject='A',
                                                                                        sen_type='active',
                                                                                        num_instances=1,
                                                                                        reps_to_use=10,
                                                                                        noMag=False,
                                                                                        sorted_inds=None)
    new_labels = [lab if len(lab) > 2 else [lab[0], lab[1], ''] for lab in labels_old]
    is_long_old = [len(lab) > 2 for lab in labels_old]
    labels_old = np.array(new_labels)
    time_old = np.squeeze(time_old)
    inds_to_plot = np.logical_and(time_old >= (tmin+0.5), time_old <=(tmax+0.5))
    print(inds_to_plot.shape)
    print(time_old.shape)
    print(data_old.shape)
    data_old = data_old[:, :, inds_to_plot]
    time_old = time_old[inds_to_plot]

    # subject, align_to, voice, experiment, proc, num_instances, reps_filter,
    # sensor_type = None, is_region_sorted = True):
    data_new, labels_new, indices_in_master_experiment_stimuli, time_new, sensor_regions= load_data_new.load_sentence_data_v2(subject='A',
                                                                                                                              align_to='noun1',
                                                                                                                              voice=('active', 'passive'),
                                                                                                                              experiment='PassAct3',
                                                                                                                              proc=None,
                                                                                                                              num_instances=1,
                                                                                                                              reps_filter=None,
                                                                                                                              sensor_type=None,
                                                                                                                              is_region_sorted=False,
                                                                                                                              tmin=tmin,
                                                                                                                              tmax=tmax)

    def num_stimulus_words(stimuli_dict_):
        return len(
            [w for w in load_data_new.punctuation_regex.sub('', stimuli_dict_['stimulus']).strip().split() if len(w.strip()) > 0])

    stimuli_annotations = list(load_data_new.read_stimuli('PassAct3'))
    assert(len(stimuli_annotations) == data_new.shape[0])
    indicator_long = np.array([num_stimulus_words(s) >= 4 for s in stimuli_annotations])
    indicator_active = np.array([s['voice'] == 'active' for s in stimuli_annotations])


    is_long_new =np.logical_and(indicator_active, indicator_long)
    time_new = np.squeeze(time_new)
    # inds_to_plot = np.logical_and(time_new >= (tmin), time_new <= (tmax))
    # data_new = data_new[:, :, inds_to_plot]
    # time_new = time_new[inds_to_plot]
    print(time_old[:10])
    print(time_new[:10])


    min_time = np.min([data_new.shape[-1], data_old.shape[-1]])
    text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
    max_line = 2.51 * 500
# for i in range(labels_new.shape[0]):
    #     for j in range(labels_old.shape[0]):
    #         if np.array_equal(labels_new[i, :], labels_old[j, :]):
    #             fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    #             h0 = ax[0].imshow(np.squeeze(data_new[i, :, :]), interpolation='nearest', aspect='auto')
    #

    #             for i_v, v in enumerate(np.arange(0.0, max_line, 0.5 * 500)):
    #                 ax[0].axvline(x=v, color='k')
    #                 if i_v < len(text_to_write):
    #                     ax[0].text(v + 0.05 * 500, 15, text_to_write[i_v])
    #             ax[0].set_title('New')
    #             h1 = ax[1].imshow(np.squeeze(data_old[j, :, :]), interpolation='nearest', aspect='auto')
    #             for i_v, v in enumerate(np.arange(0.0, max_line, 0.5 * 500)):
    #                 ax[1].axvline(x=v, color='k')
    #                 if i_v < len(text_to_write):
    #                     ax[1].text(v + 0.05 * 500, 15, text_to_write[i_v])
    #             ax[1].set_title('Old')
    #             fig.suptitle(labels_new[i, :])
    #             # fig, ax = plt.subplots()
    #             # ax.imshow(np.squeeze(np.abs(data_new[i, :, :min_time] - data_old[j, :, :min_time])), interpolation='nearest', aspect='auto')
    #             # fig.suptitle('Absolute difference')

    total_mean_new_long = np.squeeze(np.mean(data_new[is_long_new, :, :], axis=0))
    total_mean_old_long = np.squeeze(np.mean(data_old[is_long_old, :, :], axis=0))
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(total_mean_new_long, interpolation='nearest', aspect='auto')
    ax[0].set_title('New')
    ax[1].imshow(total_mean_old_long, interpolation='nearest', aspect='auto')
    ax[1].set_title('Old')
    for i_v, v in enumerate(np.arange(0.0, max_line, 0.5 * 500)):
        ax[0].axvline(x=v, color='k')
        if i_v < len(text_to_write):
            ax[0].text(v + 0.05 * 500, 15, text_to_write[i_v])
    for i_v, v in enumerate(np.arange(0.0, max_line, 0.5 * 500)):
        ax[1].axvline(x=v, color='k')
        if i_v < len(text_to_write):
            ax[1].text(v + 0.05 * 500, 15, text_to_write[i_v])
    fig.suptitle('Mean over all Long Sentences')

    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(np.abs(total_mean_new_long[:,:min_time] - total_mean_old_long[:,:min_time])),
              interpolation='nearest', aspect='auto')
    for i_v, v in enumerate(np.arange(0.0, max_line, 0.5 * 500)):
        ax.axvline(x=v, color='w')
        if i_v < len(text_to_write):
            ax.text(v + 0.05 * 500, 15, text_to_write[i_v], color='w')
    ax.set_title('Absolute Difference - Long Sentences')

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(total_mean_new_long, interpolation='nearest', aspect='auto')
    ax[0].set_title('New')
    ax[1].imshow(total_mean_old_long[:, ::2], interpolation='nearest', aspect='auto')
    ax[1].set_title('Old')
    for i_v, v in enumerate(np.arange(0.0, max_line/2.0, 0.25 * 500)):
        ax[0].axvline(x=v, color='k')
        if i_v < len(text_to_write):
            ax[0].text(v + 0.05 * 500, 15, text_to_write[i_v])
    for i_v, v in enumerate(np.arange(0.0, max_line/2.0, 0.25 * 500)):
        ax[1].axvline(x=v, color='k')
        if i_v < len(text_to_write):
            ax[1].text(v + 0.05 * 500, 15, text_to_write[i_v])
    fig.suptitle('Mean over all Long Sentences\nDownsample Test')

    plt.show()