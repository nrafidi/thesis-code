function test_pvals()
    num_perms = 100;
    perm_accs = randn(num_perms, 10, 2, 200) + 0.5;

    for goodness = [0.9]
        good_points = goodness*ones(10, 2, 100);
        for badness = 0.5:0.01:0.6
            bad_points = badness*ones(10, 2, 100);
            true_accs = cat(3, good_points, bad_points);
            disp(size(true_accs))
            rep_true_accs = repmat(true_accs, [1, 1, 1, num_perms]);
            disp(size(rep_true_accs))
            rep_true_accs = permute(rep_true_accs, [4, 1, 2, 3]);
            meow = squeeze(sum(perm_accs >= rep_true_accs, 1));
            uncorr_pvals = (meow + 1)/num_perms;
            disp(size(uncorr_pvals))
            new_pvals = correct_pvals(uncorr_pvals);
            figure
            imagesc(new_pvals)
            suptitle(sprintf('badness = %0.3f, goodness = %0.3f', badness, goodness))
            colorbar
            caxis([0, 0.05])
        end
    end
end

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
            
%             if t_stat < 0.0
%                 new_pvals(i, j) = new_pvals(i, j)/2.0;
%             else
%                 new_pvals(i, j) = 1.0 - new_pvals(i, j)/2.0;
%             end
        end
    end
end


