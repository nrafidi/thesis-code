import numpy as np
import sklearn.linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
from numpy.random import rand
import matplotlib
matplotlib.use('TkAgg') # TkAgg - only works when sshing from office machine
import matplotlib.pyplot as plt

WIN_LEN_OPTIONS = [12, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
NUM_FEAT_OPTIONS = [range(25, 500, 25), range(500, 2000, 100), range(2000, 40000, 1000)]
ALPHAS_RIDGE = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
ENET_RATIOS = [.1, .5, .7, .9, .95, .99, 1]

def flatten_list(list_of_lists):
    lst = []
    for item in list_of_lists:
        lst.extend(item)
    return lst


def lin_reg(brain_data,
            semantic_vectors,
            l_ints,
            kf,
            reg='ridge',
            adj='zscore',
            ddof=1):

    n_tot = brain_data.shape[0]
    data = np.reshape(brain_data, (n_tot, -1))

    preds = []
    test_data_all = []
    cv_membership = []
    i_split = 0
    for in_train, in_test in kf.split(data, l_ints):
        print(i_split)
        i_split += 1
        cv_membership.append(in_test)

        train_data = data[in_train, :]
        train_vectors = semantic_vectors[in_train, :]

        test_data = data[in_test, :]
        test_data_all.append(test_data)
        test_vectors = semantic_vectors[in_test, :]

        if adj == 'mean_center':
            mu_train = np.mean(train_vectors, axis=0)
            train_vectors -= mu_train[None, :]
            test_vectors -= mu_train[None, :]
            fit_intercept=False
        elif adj == 'zscore':
            mu_train = np.mean(train_vectors, axis=0)
            std_train = np.std(train_vectors, axis=0, ddof=ddof)
            train_vectors -= mu_train[None, :]
            test_vectors -= mu_train[None, :]
            train_vectors /= std_train[None, :]
            test_vectors /= std_train[None, :]
            fit_intercept = False
        else:
            fit_intercept = True

        if reg == 'ols':
            model = sklearn.linear_model.LinearRegression(fit_intercept=fit_intercept)
        elif reg == 'ridge':
            model = sklearn.linear_model.RidgeCV(alphas=ALPHAS_RIDGE, fit_intercept=fit_intercept)
        elif reg == 'lasso':
            model = sklearn.linear_model.MultiTaskLassoCV(fit_intercept=fit_intercept, max_iter=100)
        elif reg == 'enet':
            model = sklearn.linear_model.MultiTaskElasticNetCV(l1_ratio=ENET_RATIOS, fit_intercept=fit_intercept, max_iter=100)
        else:
            raise NameError('Algorithm not implemented')

        model.fit(train_vectors, train_data)

        preds.append(model.predict(test_vectors))

    preds = np.concatenate(preds, axis=0)
    print(preds.shape)
    test_data_all = np.concatenate(test_data_all, axis=0)
    print(test_data_all.shape)

    scores = explained_variance_score(test_data_all, preds, multioutput='raw_values')

    return preds, l_ints, cv_membership, scores, test_data_all



def gnb_model(data, label_membership, ddof):
    mu_full = np.array([np.mean(data[label_in, ...], axis=0) for label_in in label_membership])
    std_full = np.array([np.std(data[label_in, ...], axis=0, ddof=ddof) for label_in in label_membership])

    std_full = np.mean(std_full, axis=0)
    double_var_full = 2 * np.square(std_full)

    B = np.divide(np.square(mu_full), double_var_full[None, ...])
    A = (2 * mu_full) / double_var_full[None, ...]
    return A, B, mu_full


def get_pred_acc(preds, labels):
    label_est = np.argmax(preds, axis=1)
    return np.sum(label_est == labels)/len(labels)


def nb_tgm(data,
           labels,
           kf,
           sub_rs,
           win_starts,
           win_len,
           feature_select=False,
           doZscore=False,
           doAvg=False,
           ddof=1):

    labels = np.array(labels)
    n_tot = data.shape[0]
    n_time = data.shape[2]

    l_set = np.unique(labels)
    n_l = len(l_set)
    l_index = {l_set[i]: i for i in xrange(n_l)}
    l_ints = np.array([l_index[l] for l in labels])
    in_l = [l_ints == i for i in xrange(n_l)]

    test_windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
    n_w = len(test_windows)

    preds = np.empty((kf.get_n_splits(), n_w, n_w), dtype=np.object)
    cv_membership = np.empty((kf.get_n_splits(),), dtype=np.object)
    feature_masks = np.empty((kf.get_n_splits(), n_w), dtype=np.object)
    num_feat_selected = np.empty((kf.get_n_splits(), n_w))
    # Top-level CV
    i_top_split = 0
    for in_train, in_test in kf.split(np.reshape(data, (n_tot, -1)), l_ints):
        sub_kf = KFold(n_splits=len(in_train), shuffle=True, random_state=sub_rs)
        cv_membership[i_top_split] = in_test
        # Iterate over full time grid
        for wi in xrange(n_w):
            train_time = test_windows[wi]
            train_data = data[in_train, :, :]
            train_data = train_data[:, :, train_time]
            print(train_data.shape)
            if doAvg:
                train_data = np.mean(train_data, axis=2)
            if doZscore:
                mu_full_all = np.mean(train_data, axis=0)
                std_full_all = np.std(train_data, axis=0, ddof=ddof)
                train_data_z = train_data - mu_full_all[None, ...]
                train_data_z /= std_full_all[None, ...]
            else:
                train_data_z = train_data

            train_l_ints = l_ints[in_train]
            train_in_l = [in_l[li][in_train] for li in xrange(n_l)]

            A_top, B_top, mu_full_top = gnb_model(train_data_z, train_in_l, ddof)

            if feature_select:
                num_feat_opts = flatten_list(NUM_FEAT_OPTIONS)
                num_feat_accs = np.empty((len(num_feat_opts), sub_kf.get_n_splits()))
                # Inner CV for Feature Selection
                meow = np.reshape(train_data, (train_data.shape[0], -1))
                i_split = 0
                for in_sub_train, in_sub_test in sub_kf.split(meow, train_l_ints):
                    sub_train_data = train_data[in_sub_train, ...]
                    sub_test_data = train_data[in_sub_test, ...]
                    sub_test_ints = train_l_ints[in_sub_test]
                    sub_train_in_l = [train_in_l[li][in_sub_train] for li in xrange(n_l)]

                    if doZscore:
                        mu_full_all = np.mean(sub_train_data, axis=0)
                        std_full_all = np.std(sub_train_data, axis=0, ddof=ddof)
                        sub_train_data -= mu_full_all[None, ...]
                        sub_train_data /= std_full_all[None, ...]
                        sub_test_data -= mu_full_all[None, ...]
                        sub_test_data /= std_full_all[None, ...]
                    # print('{}/{}'.format(wi, n_w))
                    # print(sub_train_in_l)
                    A, B, mu_full = gnb_model(sub_train_data, sub_train_in_l, ddof)
                    mu_diff = reduce(lambda accum, lis: accum + np.abs(mu_full[lis[0]] - mu_full[lis[1]]),
                                     ((li1, li2) for li1 in xrange(n_l) for li2 in xrange(li1 + 1, n_l)),
                                     np.zeros(mu_full[0].shape))
                    sorted_indices = np.argsort(-mu_diff, axis=None)

                    for i_feat_num, feat_num in enumerate(num_feat_opts):
                        if feat_num > len(sorted_indices):
                            break
                        mask = np.zeros(mu_diff.shape, dtype=np.bool)
                        mask[np.unravel_index(sorted_indices[:feat_num], mu_diff.shape)] = True
                        A_mask = np.multiply(mask[None, ...], A)
                        if doAvg:
                            B_mask = np.sum(np.multiply(B, mask[None, ...]), axis=1)
                        else:
                            B_mask = np.sum(np.multiply(B, mask[None, ...]), axis=(1, 2))
                        if doAvg:
                            pred_mask = np.sum(np.multiply(sub_test_data[:, None, ...], A_mask[None, ...]),
                                               axis=2) - B_mask
                        else:
                            pred_mask = np.sum(np.multiply(sub_test_data[:, None, ...], A_mask[None, ...]),
                                               axis=(2, 3)) - B_mask
                        num_feat_accs[i_feat_num, i_split] = get_pred_acc(pred_mask, sub_test_ints)
                    i_split += 1

                best_feat_ind = np.argmax(np.mean(num_feat_accs, axis=1))
                feat_num = num_feat_opts[best_feat_ind]
                mu_diff = reduce(lambda accum, lis: accum + np.abs(mu_full_top[lis[0]] - mu_full_top[lis[1]]),
                                 ((li1, li2) for li1 in xrange(n_l) for li2 in xrange(li1 + 1, n_l)),
                                 np.zeros(mu_full_top[0].shape))
                sorted_indices = np.argsort(-mu_diff, axis=None)
                mask = np.zeros(mu_diff.shape, dtype=np.bool)
                mask[np.unravel_index(sorted_indices[:feat_num], mu_diff.shape)] = True
                feature_masks[i_top_split, wi] = mask
                num_feat_selected[i_top_split, wi] = feat_num
            else:
                mask = np.ones(A_top.shape[1:], dtype=np.bool)

            A_top = np.multiply(mask[None, ...], A_top)
            if doAvg:
                B_top = np.sum(np.multiply(B_top, mask[None, ...]), axis=1)
            else:
                B_top = np.sum(np.multiply(B_top, mask[None, ...]), axis=(1, 2))

            for wj in xrange(n_w):
                test_time = test_windows[wj]
                test_data = data[in_test, :, :]
                test_data = test_data[:, :, test_time]
                if doAvg:
                    test_data = np.mean(test_data, axis=2)
                if doZscore:
                    mu_full_all = np.mean(train_data, axis=0)
                    std_full_all = np.std(train_data, axis=0, ddof=ddof)
                    test_data -= mu_full_all[None, ...]
                    test_data /= std_full_all[None, ...]
                print(test_data.shape)
                print('A')
                print(A_top.shape)
                print('B')
                print(B_top.shape)
                meow = np.multiply(test_data[:, None, ...], A_top[None, ...])
                print('meow')
                print(meow.shape)
                if doAvg:
                    pred_top = np.sum(np.multiply(test_data[:, None, ...], A_top[None, ...]),
                                       axis=2) - B_top
                else:
                    pred_top = np.sum(np.multiply(test_data[:, None, ...], A_top[None, ...]),
                                      axis=(2, 3)) - B_top
                preds[i_top_split, wi, wj] = pred_top
        i_top_split += 1
    return preds, l_ints, cv_membership, feature_masks, num_feat_selected


def nb_tgm_uni(data,
               labels,
               kf,
               win_starts,
               win_len,
               doAvg=False,
               doZscore=False,
               ddof=1):
    print('uni')
    print(data.shape)
    labels = np.array(labels)
    n_tot = data.shape[0]
    n_time = data.shape[2]

    l_set = np.unique(labels)
    n_l = len(l_set)
    l_index = {l_set[i]: i for i in xrange(n_l)}
    l_ints = np.array([l_index[l] for l in labels])
    in_l = [l_ints == i for i in xrange(n_l)]

    test_windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
    n_w = len(test_windows)
    print(n_w)

    preds = np.empty((kf.get_n_splits(), n_w, n_w), dtype=np.object)
    cv_membership = np.empty((kf.get_n_splits(),), dtype=np.object)
    # Top-level CV
    i_top_split = 0
    for in_train, in_test in kf.split(np.reshape(data, (n_tot, -1)), l_ints):
        cv_membership[i_top_split] = in_test
        # Iterate over full time grid
        for wi in xrange(n_w):
            train_time = test_windows[wi]
            train_data = data[in_train, :, :]
            train_data = train_data[:, :, train_time]
            if doAvg:
                train_data = np.mean(train_data, axis=2)
            if doZscore:
                mu_full_all = np.mean(train_data, axis=0)
                std_full_all = np.std(train_data, axis=0, ddof=ddof)
                train_data_z = train_data - mu_full_all[None, ...]
                train_data_z /= std_full_all[None, ...]
            else:
                train_data_z = train_data

            train_in_l = [in_l[li][in_train] for li in xrange(n_l)]

            A_top, B_top, mu_full_top = gnb_model(train_data_z, train_in_l, ddof)

            # mask = np.ones(A_top.shape[1:], dtype=np.bool)
            # A_top = np.multiply(mask[None, ...], A_top)
            # print(B_top.shape)
            # if doAvg:
            #     B_top = np.sum(np.multiply(B_top, mask[None, ...]), axis=1)
            # else:
            #     B_top = np.sum(np.multiply(B_top, mask[None, ...]), axis=(1, 2))
            # print(B_top.shape)
            # print(B_top)

            for wj in xrange(n_w):
                test_time = test_windows[wj]
                test_data = data[in_test, :, :]
                test_data = test_data[:, :, test_time]
                if doAvg:
                    test_data = np.mean(test_data, axis=2)
                if doZscore:
                    mu_full_all = np.mean(train_data, axis=0)
                    std_full_all = np.std(train_data, axis=0, ddof=ddof)
                    test_data -= mu_full_all[None, ...]
                    test_data /= std_full_all[None, ...]
                if doAvg:
                    pred_top = np.squeeze(
                        np.multiply(test_data[:, None, ...], A_top[None, ...]) - B_top[None, :, None])
                else:
                    # print(A_top.shape)
                    # print(B_top.shape)
                    # print(test_data.shape)
                    pred_top = np.multiply(test_data[:, None, ...], A_top[None, ...]) - B_top[None, ...] #[None, :, None, None])
                # print(pred_top.shape)
                preds[i_top_split, wi, wj] = pred_top
        i_top_split += 1
    return preds, l_ints, cv_membership


def nb_tgm_coef(data,
                labels,
                win_starts,
                win_len,
                doZscore=False,
                doAvg=False,
                ddof=1):
    labels = np.array(labels)
    n_time = data.shape[2]

    l_set = np.unique(labels)
    n_l = len(l_set)
    l_index = {l_set[i]: i for i in xrange(n_l)}
    l_ints = np.array([l_index[l] for l in labels])
    in_l = [l_ints == i for i in xrange(n_l)]

    test_windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
    n_w = len(test_windows)

    mu_win = np.empty((n_w,), dtype=np.object)
    mu_diff_win = np.empty((n_w,), dtype=np.object)
    std_win = np.empty((n_w,), dtype=np.object)
    mu_full_all = np.mean(data, axis=0)
    std_full_all = np.std(data, axis=0, ddof=ddof)
    for wi in xrange(n_w):
        train_time = test_windows[wi]
        if doZscore:
            new_data = data[:, :, train_time] - mu_full_all[None, :, train_time]
            new_data = new_data / std_full_all[None, :, train_time]
        else:
            new_data = data[:, :, train_time]
        if doAvg:
            new_data = np.mean(new_data, axis=2)
            mu_full = np.array([np.mean(
                new_data[in_l[li], :],
                0) for li in xrange(n_l)])
            std_full = np.array([np.std(
                new_data[in_l[li], :],
                axis=0, ddof=ddof) for li in xrange(n_l)])
        else:
            mu_full = np.array([np.mean(
                new_data[in_l[li], :, :],
                0) for li in xrange(n_l)])
            std_full = np.array([np.std(
                new_data[in_l[li], :, :],
                axis=0, ddof=ddof) for li in xrange(n_l)])

        std_win[wi] = np.mean(std_full, axis=0)
        mu_win[wi] = mu_full

        mu_diff_win[wi] = reduce(lambda accum, lis: accum + np.abs(mu_full[lis[0]] - mu_full[lis[1]]),
                         ((li1, li2) for li1 in xrange(n_l) for li2 in xrange(li1 + 1, n_l)),
                         np.zeros(mu_full[0].shape))

    return mu_win, std_win, mu_diff_win


def lr_tgm(data,
           labels,
           kf,
           win_starts,
           win_len,
           doZscore=False,
           doAvg=False,
           ddof=1):
    labels = np.array(labels)
    n_tot = data.shape[0]
    n_time = data.shape[2]

    l_set = np.unique(labels)
    n_l = len(l_set)
    l_index = {l_set[i]: i for i in xrange(n_l)}
    l_ints = np.array([l_index[l] for l in labels])

    test_windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
    n_w = len(test_windows)

    preds = []
    cv_membership = []
    coef = []
    for in_train, in_test in kf.split(np.reshape(data, (n_tot, -1)), l_ints):
        cv_membership.append(in_test)
        preds.append(np.empty((n_w, n_w), dtype=object))
        coef.append(np.empty((n_w,), dtype=object))
        mu_full_all = np.mean(data[in_train, :, :], axis=0)
        std_full_all = np.std(data[in_train, :, :], axis=0, ddof=ddof)

        train_labels = labels[in_train]
        train_labels_int = l_ints[in_train]
        cv_sub = np.ones((len(train_labels),), dtype=np.int)
        for l in list(set(train_labels)):
            indLabel = [i for i, x in enumerate(train_labels) if x == l]
            cv_sub[indLabel[0::2]] = 0
        in_train_sub = cv_sub == 1
        in_test_sub = np.logical_not(in_train_sub)

        for wi in xrange(n_w):
            train_time = test_windows[wi]
            if doZscore:
                new_data = data[:, :, train_time] - mu_full_all[None, :, train_time]
                new_data = new_data / std_full_all[None, :, train_time]
            else:
                new_data = data[:, :, train_time]
            if doAvg:
                new_data = np.mean(new_data, axis=2)
                c_range = [2e11, 5e11, 8e11, 1e12, 2e12, 5e12, 8e12, 1e13]
            else:
                new_data = np.reshape(new_data, [new_data.shape[0], -1], 'F')
                c_range = [1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20]

            LR_model = sklearn.linear_model.LogisticRegression(C=1e11, penalty='l1', warm_start=True)

            sub_acc = np.zeros((len(c_range),))
            for c in c_range:
                LR_model.fit(new_data[in_train_sub, :], train_labels_int[in_train_sub])
                sub_acc = LR_model.score(new_data[in_test_sub, :], train_labels_int[in_test_sub])
                LR_model.set_params(C=c)

            LR_model.set_params(C=c_range[np.argmax(sub_acc)])

            LR_model.fit(new_data[in_train, :], train_labels_int)

            coef[-1][wi] = LR_model.coef_

            test_data_full = data[in_test, :, :]
            for wj in xrange(n_w):
                test_time = test_windows[wj]
                test_data = test_data_full[:, :, test_time]
                if doZscore == 1:
                    test_data = test_data - mu_full_all[None, :, train_time]
                    test_data = test_data / std_full_all[None, :, train_time]
                if doAvg == 1:
                    test_data = np.mean(test_data, axis=2)
                else:
                    test_data = np.reshape(test_data, (test_data.shape[0], -1), 'F')
                preds[-1][wi, wj] = LR_model.predict_log_proba(test_data)

    return preds, l_ints, cv_membership, coef


def lr_tgm_coef(data, labels, win_starts, win_len, doZscore=False, ddof=1, doAvg=False):
    labels = np.array(labels)
    n_time = data.shape[2]

    l_set = np.unique(labels)
    n_l = len(l_set)
    l_index = {l_set[i]: i for i in xrange(n_l)}
    l_ints = np.array([l_index[l] for l in labels])

    test_windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
    n_w = len(test_windows)

    coef = np.empty((n_w,), dtype=object)

    mu_full_all = np.mean(data, axis=0)
    std_full_all = np.std(data, axis=0, ddof=ddof)

    cv_sub = np.ones((len(labels),), dtype=np.int)
    for l in list(set(labels)):
        indLabel = [i for i, x in enumerate(labels) if x == l]
        cv_sub[indLabel[0::2]] = 0
    in_train_sub = cv_sub == 1
    in_test_sub = np.logical_not(in_train_sub)

    for wi in xrange(n_w):
        train_time = test_windows[wi]
        if doZscore:
            new_data = data[:, :, train_time] - mu_full_all[None, :, train_time]
            new_data = new_data / std_full_all[None, :, train_time]
        else:
            new_data = data[:, :, train_time]
            if doAvg:
                new_data = np.mean(new_data, axis=2)
                c_range = [2e11, 5e11, 8e11, 1e12, 2e12, 5e12, 8e12, 1e13]
            else:
                new_data = np.reshape(new_data, [new_data.shape[0], -1], 'F')
                c_range = [1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20]

        LR_model = sklearn.linear_model.LogisticRegression(C=1e11, penalty='l1', warm_start=True)

        sub_acc = np.zeros((len(c_range),))
        for c in c_range:
            LR_model.fit(new_data[in_train_sub, :], l_ints[in_train_sub])
            sub_acc = LR_model.score(new_data[in_test_sub, :], l_ints[in_test_sub])
            LR_model.set_params(C=c)

        LR_model.set_params(C=c_range[np.argmax(sub_acc)])

        LR_model.fit(new_data, l_ints)

        coef[wi] = LR_model.coef_

    return coef


if __name__ == '__main__':
    data = rand(16, 306, 2000)
    labels = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
    semantic_vectors = rand(16, 300) + labels[:, None]

    kf = KFold(n_splits=16)
    preds, l_ints, cv_membership, scores = lin_reg(data,
                                                   semantic_vectors,
                                                   labels,
                                                   kf,
                                                   reg='ridge',
                                                   adj='zscore',
                                                   ddof=1)
    print(scores.shape)
    # scores = np.reshape(scores, (306, 2000))
    # fig, ax = plt.subplots()
    # ax.imshow(scores, interpolation='nearest', aspect='auto')
    # plt.show()
