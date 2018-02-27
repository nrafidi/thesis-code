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
                                                                                  tmax=5.0)
    new_labels = [lab if len(lab) > 2 else [lab[0], lab[1], ''] for lab in labels_old]
    labels_old = np.array(new_labels)

    label_match = np.where(labels_new == labels_old)

    for match in label_match:
        fig, ax = plt.subplots(1, 2, figsize=(10,10))
        h0 = ax[0].imshow(np.squeeze(data_new[match, :, :]), interpolation='nearest')
        h1 = ax[1].imshow(np.squeeze(data_old[match, :, :]), interpolation='nearest')
        fig.suptitle(labels_new[match, :])
    plt.show()