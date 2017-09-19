import numpy as np
import scipy.io
import hippo.query
import hippo.models
import cmumetadata.metaio as cmuio
import os.path
import string
import sklearn.linear_model
#from sklearn import linear_model
win_len = 100
timeToClass = 25
experiment = 'krns2'
subject = 'B'
proc_slug = 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'

usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['doctor', 'peach','dog','hammer','school','door','monkey','student']),   ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) 

exp_sub = [(experiment, subject)]
uels = hippo.query.get_uels_from_usis(usis.keys(), experiment_subjects=exp_sub)
uels = {k : v for (k,v) in uels.iteritems() if len(v) > 0} #checking for empties
id_uels = [(k, uels[k]) for k in uels.keys()] #putting uels in a list instead of a map (explicit ordering)
  #Labels
labels = [usis[k]['stimulus'] for k, _ in id_uels]
labels = np.array(labels)
l_set = np.unique(labels)
n_l = len(l_set)
l_index = {l_set[i] : i  for i in xrange(n_l)}
l_ints = np.array([l_index[l] for l in labels])
in_l = [l_ints == i for i in xrange(n_l)]

  #CV scheme
usi_list = [k for k,_ in id_uels]
cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))

  #2 Fold CV
cv = np.ones((len(labels),), dtype=np.int);
for l in list(set(labels)):
    indLabel = [i for i,x in enumerate(labels) if x == l]
    cv[indLabel[0::2]] = 0

in_train = cv == 1
in_test = np.logical_not(in_train)

print(l_ints)

data = np.concatenate((np.reshape(l_ints, [-1,1]),np.reshape(l_ints, [-1,1])),axis=1)

train_data = data[in_train,:]
train_labels = l_ints[in_train]
#,multi_class='multinomial'
#class_weight='balanced'
LR_model = sklearn.linear_model.LogisticRegression(C=1e-3,solver='liblinear',penalty='l2',warm_start=True, tol=1e-6, max_iter=1000)

for c in [0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]:
    LR_model.fit(train_data,train_labels)
    accuracy = LR_model.score(data[in_test,:], l_ints[in_test])
    print(accuracy)
    print(LR_model.n_iter_)
    print(np.mean(LR_model.coef_, axis=1))
    LR_model.set_params(C=c)
