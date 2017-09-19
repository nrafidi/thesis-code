import numpy as np
import scipy.io
import hippo.query
import hippo.models
import cmumetadata.metaio as cmuio
import os.path
import string
import sklearn.linear_model
import h5py

win_len = 100

wordOnset = 0.5
numSentences = 16
numReps = 15
experiment = 'krns2'
subject = 'B'
dataDir = '/share/volume0/newmeg/' + experiment + '/avg/' + subject + '/';
fname = dataDir + experiment + '_' + subject + '_' + 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas_parsed_pdtwSyn.mat'

loadVars = h5py.File(fname);

activeData = np.transpose(loadVars[u'activeData'],axes=(2, 1, 0))
fullWordLab = loadVars[u'fullWordLab']
fullTime = np.array(loadVars[u'fullTime'])


data = np.empty((numSentences*2, activeData.shape[1], activeData.shape[2]))
for s in range(0, numSentences):
    startInd = numReps*s
    endInd = startInd + numReps - 1
    data[s,:,:] = np.mean(activeData[startInd:endInd:2,:, :], axis=0)
    data[s+numSentences,:,:] = np.mean(activeData[(startInd+1):endInd:2,:, :], axis=0)

labels = fullWordLab[0,0:(numSentences-1)]
labels = labels + labels

  #2 Fold CV
cv = np.ones((len(labels),), dtype=np.int);
for l in list(set(labels)):
    indLabel = [i for i,x in enumerate(labels) if x == l]
    cv[indLabel[0::2]] = 0

tmin = -0.5 + wordOnset
tmax = 1.5 + wordOnset

print(tmin)
print(fullTime.min())
print(fullTime.max())


timeToUse = np.squeeze(np.logical_and(fullTime >= tmin,fullTime <= tmax))
print(np.sum(timeToUse))
data = data[:,:,timeToUse]
print(data.shape)

total_win = int((tmax-tmin)*500)
overlap =  12
win_starts = range(0,total_win-win_len, overlap)

n_time = data.shape[2]
test_windows = [np.array([i>=w_s and i <w_s + win_len for i in xrange(n_time)]) for w_s in win_starts]
print(len(test_windows))

for timeToClass in range(55, 75):
#    print(test_windows[timeToClass])
    # Window to test classification
    winData = np.reshape(data[:,:,test_windows[timeToClass]], newshape=[32,-1])

    in_train = cv == 1
    in_test = np.logical_not(in_train)

    train_data = winData[in_train,:]
    train_labels = labels[in_train]

    LR_model = sklearn.linear_model.LogisticRegression(C=1e11,solver='liblinear',penalty='l1',warm_start=True, tol=1e-4, max_iter=100)

    cRange = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19]
# for c in [2e11, 5e11, 8e11, 1e12, 2e12, 5e12, 8e12]
    accuracy = np.empty((len(cRange), ))
    i = 0
    for c in cRange:
        LR_model.fit(train_data,train_labels)
        accuracy[i] = LR_model.score(winData[in_test,:], labels[in_test])
        i += 1
        LR_model.set_params(C=c)
    print(accuracy.max())
    print(cRange[accuracy.argmax()])
