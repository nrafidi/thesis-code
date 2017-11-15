import numpy as np
from scipy.stats import norm
from scipy import stats
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def correct_pvals(uncorrected_pvals):
    print('moo')
    up_shape = uncorrected_pvals.shape
    print(up_shape)
    new_pvals = np.empty((uncorrected_pvals.shape[1], uncorrected_pvals.shape[2]))
    print(new_pvals.shape)
    for i in range(uncorrected_pvals.shape[1]):
        print(i)
        for j in range(uncorrected_pvals.shape[2]):
            # fig, axs = plt.subplots()

            dist_over_sub = uncorrected_pvals[:, i, j]
            # print(np.min(dist_over_sub))
            # axs[0][0].hist(dist_over_sub)
            dist_over_sub[dist_over_sub == 1.0] -= 1e-14
            dist_over_sub[dist_over_sub == 0.0] += 1e-14
            # axs[0][1].hist(dist_over_sub)
            # print(np.min(dist_over_sub))
            # print('ahoy')
            # print(dist_over_sub)
            meow = norm.ppf(dist_over_sub)
            # axs[1][0].hist(meow)
            assert not np.any(np.isinf(meow))
            assert not np.any(np.isnan(meow))
            # print(meow)
            # meow[meow == 1.0] -= 1e-14
            # meow[meow == 0.0] += 1e-14
            # axs.hist(meow)
            t_stat, new_pvals[i, j] = stats.ttest_1samp(meow, 0.0)
            my_t_stat = np.mean(meow)/(np.std(meow)/np.sqrt(meow.size))
            stddev = np.std(meow)
            if t_stat < 0.0:
                new_pvals[i, j] /= 2.0
            else:
                new_pvals[i, j] = 1.0 - new_pvals[i, j]/2.0
            assert not np.isnan(new_pvals[i, j])
            assert not np.isinf(new_pvals[i, j])
            # if j % 100 == 0:
            #     fig.suptitle('{}, {}, {}, {}'.format(t_stat, new_pvals[i, j], my_t_stat, stddev))
            #     plt.show()
            # else:
            #     plt.close()
            # plt.close()
    # fig, axs = plt.subplots()
    # h = axs.imshow(new_pvals, interpolation='nearest', aspect='auto')
    # plt.colorbar(h)
    bh_thresh = bhy_multiple_comparisons_procedure(new_pvals)
    print(bh_thresh)
    # plt.show()

    corr_pvals = new_pvals <= bh_thresh[:, None]
    return corr_pvals, new_pvals


def bhy_multiple_comparisons_procedure(uncorrected_pvalues, alpha=0.05):
    # originally from Mariya Toneva
    if len(uncorrected_pvalues.shape) == 1:
        uncorrected_pvalues = np.reshape(uncorrected_pvalues, (1, -1))

    # get ranks of all p-values in ascending order
    sorting_inds = np.argsort(uncorrected_pvalues, axis=1)
    ranks = sorting_inds + 1  # add 1 to make the ranks start at 1 instead of 0

    # calculate critical values under arbitrary dependence
    dependency_constant = np.sum(1 / ranks)
    critical_values = ranks * alpha / (uncorrected_pvalues.shape[1] * dependency_constant)

    # find largest pvalue that is <= than its critical value
    sorted_pvalues = np.empty(uncorrected_pvalues.shape)
    sorted_critical_values = np.empty(critical_values.shape)
    for i in range(uncorrected_pvalues.shape[0]):
        sorted_pvalues[i, :] = uncorrected_pvalues[i, sorting_inds[i, :]]
        sorted_critical_values[i, :] = critical_values[i, sorting_inds[i, :]]
    bh_thresh = -1.0*np.ones((sorted_pvalues.shape[0],))
    for j in range(sorted_pvalues.shape[0]):
        for i in range(sorted_pvalues.shape[1] - 1, -1, -1):  # start from the back
            if sorted_pvalues[j, i] <= sorted_critical_values[j, i]:
                if bh_thresh[j] < 0:
                    bh_thresh[j] = sorted_pvalues[j, i]
                    print('threshold for row ', j, ' is:', bh_thresh[j], 'critical value:', sorted_critical_values[j, i], i)

    return bh_thresh


if __name__ == '__main__':
    perm_accs = np.random.rand(1000, 10, 2, 200)

    for goodness in [0.9, 0.99]:
        good_points = goodness*np.ones((10, 2, 100))
        for badness in [0.5, 0.6, 0.7]:
            bad_points = badness*np.ones((10, 2, 100))
            true_accs = np.concatenate([good_points, bad_points], axis=2)
            print(true_accs.shape)
            # fig, axs = plt.subplots()
            # h = axs.imshow(true_accs[0, ...], interpolation='nearest', aspect='auto')
            # plt.colorbar(h)
            uncorr_pvals = np.mean(perm_accs >= true_accs[None, ...], axis=0)
            # fig, axs = plt.subplots()
            # h = axs.imshow(uncorr_pvals[0, ...], interpolation='nearest', aspect='auto')
            # plt.colorbar(h)
            corr_pvals, new_pvals = correct_pvals(uncorr_pvals)
            fig, axs = plt.subplots()
            h = axs.imshow(new_pvals, interpolation='nearest', aspect='auto')
            fig.suptitle('badness = {}, goodness = {}'.format(badness, goodness))
            plt.colorbar(h)
            print('badness = {}, goodness = {}'.format(badness, goodness))
            for i in range(2):
                ktau, _ = kendalltau(uncorr_pvals[i, ...], new_pvals)
                print(ktau)

    plt.show()
