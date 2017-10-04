import numpy as np
import sklearn.linear_model

WIN_LEN_OPTIONS = [12, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


# def nb_diag(data,
#             labels,
#             kf,
#             sub_kf,
#             win_starts,
#             feature_select=None,
#             feature_select_params=None,
#             doZscore=False,
#             doAvg=False,
#             ddof=1):
#
#     labels = np.array(labels)
#     n_tot = data.shape[0]
#     n_time = data.shape[2]
#
#     l_set = np.unique(labels)
#     n_l = len(l_set)
#     l_index = {l_set[i]: i for i in xrange(n_l)}
#     l_ints = np.array([l_index[l] for l in labels])
#     in_l = [l_ints == i for i in xrange(n_l)]
#
#     total_win = np.max(win_starts)
#
#     preds = []
#     cv_membership = []
#     feature_masks = []
#     for in_train, in_test in kf.split(np.reshape(data, (n_tot, -1)), l_ints):
#         train_data = data[in_train, :, :]
#         train_labels = l_ints
#         for w_s in win_starts:
#             for in_sub_train, in_sub_test in sub_kf.split
#         test_windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts if
#                         w_s < (total_win - win_len)]
#         n_w = len(test_windows)
#         cv_membership.append(in_test)
#         preds.append(np.zeros((n_w,), dtype=np.object))
#         feature_masks.append(np.zeros((n_w,), dtype=np.object))
#         mu_full_all = np.mean(data[in_train, :, :], axis=0)
#         std_full_all = np.std(data[in_train, :, :], axis=0, ddof=ddof)
#         for wi in xrange(n_w):
#             train_time = test_windows[wi]
#             if doZscore:
#                 new_data = data[:, :, train_time] - mu_full_all[None, :, train_time]
#                 new_data = new_data / std_full_all[None, :, train_time]
#             else:
#                 new_data = data[:, :, train_time]
#             if doAvg:
#                 new_data = np.mean(new_data, axis=2)
#                 mu_full = np.array([np.mean(
#                     new_data[np.logical_and(in_train, in_l[li]), :],
#                     0) for li in xrange(n_l)])
#                 std_full = np.array([np.std(
#                     new_data[np.logical_and(in_train, in_l[li]), :],
#                     axis=0, ddof=ddof) for li in xrange(n_l)])
#             else:
#                 mu_full = np.array([np.mean(
#                     new_data[np.logical_and(in_train, in_l[li]), :, :],
#                     0) for li in xrange(n_l)])
#                 std_full = np.array([np.std(
#                     new_data[np.logical_and(in_train, in_l[li]), :, :],
#                     axis=0, ddof=ddof) for li in xrange(n_l)])
#
#             std_full = np.mean(std_full, axis=0)
#             double_var_full = 2 * np.square(std_full)
#
#             if doAvg:
#                 B_full = np.divide(np.square(mu_full), double_var_full[None, :])
#                 A = (2 * mu_full) / double_var_full[None, :]
#             else:
#                 B_full = np.divide(np.square(mu_full), double_var_full[None, :, :])
#                 A = (2 * mu_full) / double_var_full[None, :, :]
#
#             if feature_select is not None:
#                 if feature_select == 'distance_of_means':
#                     mu_diff = reduce(lambda accum, lis: accum + np.abs(mu_full[lis[0]] - mu_full[lis[1]]),
#                                      ((li1, li2) for li1 in xrange(n_l) for li2 in xrange(li1 + 1, n_l)),
#                                      np.zeros(mu_full[0].shape))
#                     sorted_indices = np.argsort(-mu_diff, axis=None)
#                     nfeatures = feature_select_params['number_of_features']
#                     mask = np.zeros(mu_diff.shape, dtype=np.bool)
#                     mask[np.unravel_index(sorted_indices[0:nfeatures], mu_diff.shape)] = True
#                 else:
#                     raise ValueError('Invalid feature selection')
#             else:
#                 if doAvg:
#                     mask = np.ones(A[:, :].shape[1:], dtype=np.bool)
#                 else:
#                     mask = np.ones(A[:, :, :].shape[1:], dtype=np.bool)
#             feature_masks[-1][wi] = mask
#
#             # This is right, right?
#             if doAvg:
#                 A = np.multiply(mask[None, :], A)
#                 B = np.sum(np.multiply(B_full, mask[None, :]), axis=1)  # n classes x n time
#             else:
#                 A = np.multiply(mask[None, :, :], A)
#                 B = np.sum(np.multiply(B_full, mask[None, :, :]), axis=(1, 2))  # n classes x n time
#
#             P_cGx = np.empty([len(test_windows), np.sum(in_test), n_l])
#             test_data_full = data[in_test, :, :]
#             for wj in xrange(n_w):
#                 test_time = test_windows[wj]
#                 test_data = test_data_full[:, :, test_time]
#                 if doZscore:
#                     test_data = test_data - mu_full_all[None, :, train_time]
#                     test_data = test_data / std_full_all[None, :, train_time]
#                 if doAvg:
#                     test_data = np.mean(test_data, axis=2)
#                     print(test_data.shape)
#                     P_cGx[wj, :, :] = np.sum(np.multiply(test_data[:, None, :], A[None, :, :]), axis=2) - B
#                 else:
#                     P_cGx[wj, :, :] = np.sum(np.multiply(test_data[:, None, :, :], A[None, :, :, :]), axis=(2, 3)) - B
#             preds[-1][wi] = P_cGx
#     return preds, l_ints, cv_membership, feature_masks


def nb_tgm(data,
           labels,
           kf,
           win_starts,
           win_len,
           feature_select=None,
           feature_select_params=None,
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
    print(l_ints.shape)
    print(n_tot)
    in_l = [l_ints == i for i in xrange(n_l)]

    test_windows = [np.array([i >= w_s and i < w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
    n_w = len(test_windows)

    preds = []
    cv_membership = []
    feature_masks = []
    for in_train, in_test in kf.split(np.reshape(data, (n_tot, -1)), l_ints):
        cv_membership.append(in_test)
        preds.append(np.zeros((n_w,), dtype=np.object))
        feature_masks.append(np.zeros((n_w,), dtype=np.object))
        mu_full_all = np.mean(data[in_train, :, :], axis=0)
        std_full_all = np.std(data[in_train, :, :], axis=0, ddof=ddof)
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
                    new_data[np.logical_and(in_train, in_l[li]), :],
                    0) for li in xrange(n_l)])
                std_full = np.array([np.std(
                    new_data[np.logical_and(in_train, in_l[li]), :],
                    axis=0, ddof=ddof) for li in xrange(n_l)])
            else:
                mu_full = np.array([np.mean(
                    new_data[np.logical_and(in_train, in_l[li]), :, :],
                    0) for li in xrange(n_l)])
                std_full = np.array([np.std(
                    new_data[np.logical_and(in_train, in_l[li]), :, :],
                    axis=0, ddof=ddof) for li in xrange(n_l)])

            std_full = np.mean(std_full, axis=0)
            double_var_full = 2 * np.square(std_full)

            if doAvg:
                B_full = np.divide(np.square(mu_full), double_var_full[None, :])
                A = (2 * mu_full) / double_var_full[None, :]
            else:
                B_full = np.divide(np.square(mu_full), double_var_full[None, :, :])
                A = (2 * mu_full) / double_var_full[None, :, :]

            if feature_select is not None:
                if feature_select == 'distance_of_means':
                    mu_diff = reduce(lambda accum, lis: accum + np.abs(mu_full[lis[0]] - mu_full[lis[1]]),
                                     ((li1, li2) for li1 in xrange(n_l) for li2 in xrange(li1 + 1, n_l)),
                                     np.zeros(mu_full[0].shape))
                    sorted_indices = np.argsort(-mu_diff, axis=None)
                    nfeatures = feature_select_params['number_of_features']
                    mask = np.zeros(mu_diff.shape, dtype=np.bool)
                    mask[np.unravel_index(sorted_indices[0:nfeatures], mu_diff.shape)] = True
                else:
                    raise ValueError('Invalid feature selection')
            else:
                if doAvg:
                    mask = np.ones(A[:, :].shape[1:], dtype=np.bool)
                else:
                    mask = np.ones(A[:, :, :].shape[1:], dtype=np.bool)
            feature_masks[-1][wi] = mask

            # This is right, right?
            if doAvg:
                A = np.multiply(mask[None, :], A)
                B = np.sum(np.multiply(B_full, mask[None, :]), axis=1)  # n classes x n time
            else:
                A = np.multiply(mask[None, :, :], A)
                B = np.sum(np.multiply(B_full, mask[None, :, :]), axis=(1, 2))  # n classes x n time

            P_cGx = np.empty([len(test_windows), np.sum(in_test), n_l])
            test_data_full = data[in_test, :, :]
            for wj in xrange(n_w):
                test_time = test_windows[wj]
                test_data = test_data_full[:, :, test_time]
                if doZscore:
                    test_data = test_data - mu_full_all[None, :, train_time]
                    test_data = test_data / std_full_all[None, :, train_time]
                if doAvg:
                    test_data = np.mean(test_data, axis=2)
                    print(test_data.shape)
                    P_cGx[wj, :, :] = np.sum(np.multiply(test_data[:, None, :], A[None, :, :]), axis=2) - B
                else:
                    P_cGx[wj, :, :] = np.sum(np.multiply(test_data[:, None, :, :], A[None, :, :, :]), axis=(2, 3)) - B
            preds[-1][wi] = P_cGx
    return preds, l_ints, cv_membership, feature_masks


def nb_tgm_coef(data,
                labels,
                win_starts,
                win_len,
                feature_select,
                feature_select_params,
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

    feature_masks = np.empty((n_w,), dtype=np.object)
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

        std_full = np.mean(std_full, axis=0)
        double_var_full = 2 * np.square(std_full)

        if doAvg:
            A = (2 * mu_full) / double_var_full[None, :]
        else:
            A = (2 * mu_full) / double_var_full[None, :, :]

        if feature_select is not None:
            if feature_select == 'distance_of_means':
                mu_diff = reduce(lambda accum, lis: accum + np.abs(mu_full[lis[0]] - mu_full[lis[1]]),
                                 ((li1, li2) for li1 in xrange(n_l) for li2 in xrange(li1 + 1, n_l)),
                                 np.zeros(mu_full[0].shape))
                sorted_indices = np.argsort(-mu_diff, axis=None)
                nfeatures = feature_select_params['number_of_features']
                mask = np.zeros(mu_diff.shape, dtype=np.bool)
                mask[np.unravel_index(sorted_indices[0:nfeatures], mu_diff.shape)] = True
            else:
                raise ValueError('Invalid feature selection')
        else:
            if doAvg:
                mask = np.ones(A[:, :].shape[1:], dtype=np.bool)
            else:
                mask = np.ones(A[:, :, :].shape[1:], dtype=np.bool)
        feature_masks[wi] = mask

    return feature_masks


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

    cv_sub = np.ones((len(labels),), dtype=np.int);
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
