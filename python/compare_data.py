import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import load_data_ordered as load_data


if __name__ == '__main__':
    data_new, labels_new, time_new, final_inds_new = load_data.load_PassAct3_matlab(subject='A',
                                                                                    sen_type='active',
                                                                                    num_instances=1,
                                                                                    reps_to_use=10,
                                                                                    noMag=False,
                                                                                    sorted_inds=None)
    new_labels = [lab if len(lab) > 2 else [lab[0], lab[1], ''] for lab in labels_new]
    is_long_new = [len(lab) > 2 for lab in labels_new]
    labels_new = np.array(new_labels)

    data_old, labels_old, time_old, final_inds_old = load_data.load_sentence_data(subject='A',
                                                                                  word='noun1',
                                                                                  sen_type='active',
                                                                                  experiment='PassAct3',
                                                                                  proc=load_data.DEFAULT_PROC,
                                                                                  num_instances=1,
                                                                                  reps_to_use=10,
                                                                                  noMag=False,
                                                                                  sorted_inds=None,
                                                                                  tmin=-0.5,
                                                                                  tmax=4.0)
    new_labels = [lab if len(lab) > 2 else [lab[0], lab[1], ''] for lab in labels_old]
    is_long_old = [len(lab) > 2 for lab in labels_old]
    labels_old = np.array(new_labels)

    min_time = np.min([data_new.shape[-1], data_old.shape[-1]])

    for i in range(labels_new.shape[0]):
        for j in range(labels_old.shape[0]):
            if np.array_equal(labels_new[i, :], labels_old[j, :]):
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                h0 = ax[0].imshow(np.squeeze(data_new[i, :, :]), interpolation='nearest', aspect='auto')
                text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
                max_line = 2.51 * 500
                for i_v, v in enumerate(np.arange(0.5 * 500, max_line, 0.5 * 500)):
                    ax[0].axvline(x=v, color='k')
                    if i_v < len(text_to_write):
                        plt.text(v + 0.05 * 500, 15, text_to_write[i_v])
                ax[0].set_title('New')
                h1 = ax[1].imshow(np.squeeze(data_old[j, :, :]), interpolation='nearest', aspect='auto')
                for i_v, v in enumerate(np.arange(0.5 * 500, max_line, 0.5 * 500)):
                    ax[1].axvline(x=v, color='k')
                    if i_v < len(text_to_write):
                        plt.text(v + 0.05 * 500, 15, text_to_write[i_v])
                ax[1].set_title('Old')
                fig.suptitle(labels_new[i, :])
                # fig, ax = plt.subplots()
                # ax.imshow(np.squeeze(np.abs(data_new[i, :, :min_time] - data_old[j, :, :min_time])), interpolation='nearest', aspect='auto')
                # fig.suptitle('Absolute difference')

    total_mean_new_long = np.squeeze(np.mean(data_new[is_long_new, :, :], axis=0))
    total_mean_old_long = np.squeeze(np.mean(data_old[is_long_old, :, :], axis=0))
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(total_mean_new_long, interpolation='nearest', aspect='auto')
    ax[0].set_title('New')
    ax[1].imshow(total_mean_old_long, interpolation='nearest', aspect='auto')
    ax[1].set_title('Old')

    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(np.abs(total_mean_new_long[:,:min_time] - total_mean_old_long[:,:min_time])),
              interpolation='nearest', aspect='auto')
    ax.set_title('Absolute Difference - Long Sentences')
    plt.show()