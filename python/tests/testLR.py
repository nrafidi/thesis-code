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
timeToClass = 30
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

  #CV scheme
usi_list = [k for k,_ in id_uels]
cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))

  #2 Fold CV
cv = np.ones((len(labels),), dtype=np.int);
for l in list(set(labels)):
    indLabel = [i for i,x in enumerate(labels) if x == l]
    cv[indLabel[0::2]] = 0

tmin = -0.5
tmax = 1

total_win = int((tmax-tmin)*500)
overlap =  12
win_starts = range(0,total_win-win_len, overlap)



  #List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
evokeds = hippo.models.load_data(id_uels, tmin,tmax,proc_slug,baseline=None)
data = np.array([e.data for e in evokeds])

n_time = data.shape[2]
test_windows = [np.array([i>=w_s and i <w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]

l_set = np.unique(labels)
n_l = len(l_set)
l_index = {l_set[i] : i  for i in xrange(n_l)}
l_ints = np.array([l_index[l] for l in labels])
in_l = [l_ints == i for i in xrange(n_l)]

  #Downsample data
data = data[:,:,0::2]

# Window to test classification
#data = np.reshape(data[:,:,test_windows[timeToClass]], newshape=[32,-1])
data = np.mean(data[:,:,test_windows[timeToClass]], axis=2)
print(data.shape)

in_train = cv == 1
in_test = np.logical_not(in_train)

train_data = data[in_train,:]
train_labels = l_ints[in_train]
#,multi_class='multinomial'
#class_weight='balanced'
LR_model = sklearn.linear_model.LogisticRegression(C=1e11,solver='liblinear',penalty='l1',warm_start=True, tol=1e-4, max_iter=100)

print(len(win_starts))
  #[1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20]
for c in [2e11, 5e11, 8e11, 1e12, 2e12, 5e12, 8e12]:
    LR_model.fit(train_data,train_labels)
    accuracy = LR_model.score(data[in_test,:], l_ints[in_test])
    print(accuracy)
   # print(LR_model.n_iter_)
  #  print(float(np.count_nonzero(LR_model.coef_))/LR_model.coef_.size)
    LR_model.set_params(C=c)
