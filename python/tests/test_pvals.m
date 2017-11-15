function new_pvals = correct_pvals(uncorrected_pvals)
    
    [~, num_reg, num_t] = size(uncorrected_pvals);
    new_pvals =nan(num_reg, num_t);
    for i = 1:num_reg
        for j = 1:num_t
            dist_over_sub = uncorrected_pvals(:, i, j);
            dist_over_sub(dist_over_sub == 1.0) = dist_over_sub(dist_over_sub == 1.0) - 1e-14;
            dist_over_sub(dist_over_sub == 0.0) = dist_over_sub(dist_over_sub == 0.0) + 1e-14;
            
            meow = norminv(dist_over_sub);
            
            [~, new_pvals(i, j)] = ttest(meow);
            
            if t_stat < 0.0
                new_pvals(i, j) = new_pvals(i, j)/2.0;
            else
                new_pvals(i, j) = 1.0 - new_pvals(i, j)/2.0;
            end
        end
    end
end


function test_pvals
    perm_accs = np.random.rand(1000, 10, 2, 200)

    for goodness in [0.9, 0.99, 0.999]:
        good_points = goodness*np.ones((10, 2, 100))
        for badness in [0.8]:
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

                end
                end