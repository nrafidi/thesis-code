import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from syntax_vs_semantics import load_data
import numpy as np


if __name__ == '__main__':
    data, labels, sen_ints, time, sensor_regions = load_data.load_sentence_data_v2(subject='B',
                                                                                   align_to='verb',
                                                                                   voice='active',
                                                                                   experiment='krns2',
                                                                                   proc=load_data.DEFAULT_PROC,
                                                                                   num_instances=1,
                                                                                   reps_filter=None,
                                                                                   sensor_type=None,
                                                                                   is_region_sorted=False,
                                                                                   tmin=None,
                                                                                   tmax=None)
    data_to_plot = np.squeeze(np.mean(data, axis=0))

    fig, ax = plt.subplots()
    h = ax.imshow(data_to_plot, interpolation='nearest')
    plt.show()
