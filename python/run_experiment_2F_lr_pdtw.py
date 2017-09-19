import scipy.io
import argparse
import hippo.query
import hippo.TGM_models
import hippo.models
import cmumetadata.metaio as cmuio
import numpy as np
import random
import warnings
import string
import os.path
import h5py

# Runs the TGM experiment, 2F CV, pooling active and passive sentences
def pooled_krns2_lr_pdtw(experiment,subject, proc_slug,word,win_len,actORpass='pooled',doZscore=False,ddof=1,doAvg=False,do2Samp=False):

  numSentences = 32
  numReps = 15
  dataDir = '/share/volume0/newmeg/' + experiment + '/avg/' + subject + '/'
  fname = '{dataDir}{experiment}_{sub}_trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas_parsed_{word}_pdtwSyn.mat'
  fname = fname.format(dataDir=dataDir,experiment=experiment,sub=subject,word=word)

  loadVars = h5py.File(fname)
  time = np.array(loadVars[u'fullTime_p'])
  fullWordLab = loadVars[u'fullWordLab']

  numTime = time.size

# Gets the stimulus IDs for all the first nouns

  active_data_raw = np.transpose(loadVars[u'activeData'],axes=(2,1,0))
  active_data_raw = active_data_raw[:,:,0:numTime]
  passive_data_raw = np.transpose(loadVars[u'passiveData'],axes=(2,1,0))
  sensToUse = range(0, numSentences)
  if word == 'firstNoun':
    labels = fullWordLab[0, sensToUse]
    #wordOnset_a = 0.5
   # wordOnset_p = 0.5
  elif word == 'verb':
    labels = fullWordLab[1, sensToUse]
   # wordOnset_a = 1
  #  wordOnset_p = 1.5
  else:
    labels = fullWordLab[2, sensToUse]
 #   wordOnset_a = 2
#    wordOnset_p = 3

  tmin = time.min()
  tmax = time.max()

  total_win = int((tmax-tmin)*500)
  overlap =  12
  win_starts = range(0,total_win-win_len, overlap)

#  print(active_data_raw.shape[1])
 # print(np.count_nonzero(timeToUse_a))

  #Average Data
  if do2Samp == True:
    labels = labels + labels
    data = np.empty((numSentences*2, active_data_raw.shape[1], active_data_raw.shape[2]))
    for s in range(0, numSentences/2):
      startInd = numReps*s
      endInd = startInd + numReps - 1
      data[s,:,:] = np.mean(active_data_raw[startInd:endInd:2,:, :], axis=0)
      data[s+numSentences,:,:] = np.mean(active_data_raw[(startInd+1):endInd:2,:, :], axis=0)
    for s in range(0, numSentences/2):
      startInd = numReps*s
      endInd = startInd + numReps - 1
      data[s+numSentences/2,:,:] = np.mean(passive_data_raw[startInd:endInd:2,:, :], axis=0)
      data[s+(numSentences*3)/2,:,:] = np.mean(passive_data_raw[(startInd+1):endInd:2,:, :], axis=0)
  else:
    data = np.empty((numSentences, active_data_raw.shape[1], active_data_raw.shape[2]))
    for s in range(0, numSentences/2):
      startInd = numReps*s
      endInd = startInd + numReps - 1
      data[s,:,:] = np.mean(active_data_raw[startInd:endInd,:, :], axis=0)
    for s in range(0, numSentences/2):
      startInd = numReps*s
      endInd = startInd + numReps - 1
      data[s+numSentences/2,:,:] = np.mean(passive_data_raw[startInd:endInd,:, :], axis=0)

  #2 Fold CV
  cv = np.ones((len(labels),), dtype=np.int);
  for l in list(set(labels)):
    indLabel = [i for i,x in enumerate(labels) if x == l]
    cv[indLabel[0::2]] = 0

  #Run TGM
  preds,tgm,l_index,cv_membership,masks = hippo.models._lr_tgm(data, labels, cv, win_starts, win_len,doZscore,ddof,doAvg)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  fnameStub = subDir + '/TGM_{sub}_{word}_pooled_win{win_len}_zscore{doZ}_avg{doAvg}_2Samp{do2Samp}_2F_LR_PDTW.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,word=word,win_len=win_len,doZ=doZscore,doAvg=doAvg,do2Samp=do2Samp), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks, 'labels':labels,'cv_membership':cv_membership})

# Pooled
###################################################################################################################################
# Runs the TGM experiment, 2F CV, separately for active and passive sentences
def actORpass_krns2_lr_pdtw(experiment,subject, proc_slug,word,win_len,actORpass,doZscore=True,ddof=1,doAvg=False):
  warnings.filterwarnings('error')

  numSentences = 16
  numReps = 15
  dataDir = '/share/volume0/newmeg/' + experiment + '/avg/' + subject + '/'
  fname = '{dataDir}{experiment}_{sub}_trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas_parsed_{word}_pdtwSyn.mat'
  fname = fname.format(dataDir=dataDir,experiment=experiment,sub=subject,word=word)

  loadVars = h5py.File(fname)

  fullWordLab = loadVars[u'fullWordLab']

# Gets the stimulus IDs for all the first nouns
  if actORpass == 'active':
    data_raw = np.transpose(loadVars[u'activeData'],axes=(2,1,0))
    time = np.array(loadVars[u'fullTime_a'])
    sensToUse = range(0, numSentences)
    if word == 'firstNoun':
      labels = fullWordLab[0, sensToUse]
#      wordOnset = 0.5
    elif word == 'verb':
      labels = fullWordLab[1, sensToUse]
#      wordOnset = 1
    else:
      labels = fullWordLab[2, sensToUse]
#      wordOnset = 2
  else:
    data_raw = np.transpose(loadVars[u'passiveData'],axes=(2,1,0))
    time = np.array(loadVars[u'fullTime_p'])
    sensToUse = range(numSentences, numSentences*2)
    if word == 'firstNoun':
      labels = fullWordLab[0, sensToUse]
#      wordOnset = 0.5
    elif word == 'verb':
      labels = fullWordLab[1, sensToUse]
      #wordOnset = 1.5
    else:
      labels = fullWordLab[2, sensToUse]
#      wordOnset = 3

  #Average Data
  data = np.empty((numSentences*2, data_raw.shape[1], data_raw.shape[2]))
  for s in range(0, numSentences):
    startInd = numReps*s
    endInd = startInd + numReps - 1
    data[s,:,:] = np.mean(data_raw[startInd:endInd:2,:, :], axis=0)
    data[s+numSentences,:,:] = np.mean(data_raw[(startInd+1):endInd:2,:, :], axis=0)

  #Labels
  labels = labels + labels #appends labels to itself

  #2 Fold CV
  cv = np.ones((len(labels),), dtype=np.int);
  for l in list(set(labels)):
    indLabel = [i for i,x in enumerate(labels) if x == l]
    cv[indLabel[0::2]] = 0

 #Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
 # tmin = wordOnset - 0.5
 # if word == 'firstNoun':
 #   tmax = wordOnset + 4.5
 # elif word == 'verb':
 #   tmax = wordOnset + 4
 # else:
#    tmax = wordOnset + 3
  
#  if tmin < time.min():
  tmin = time.min()
#  if tmax > time.max():
  tmax = time.max()

  overlap = 12;

  #timeToUse = np.squeeze(np.logical_and(time >= tmin,time < tmax))
  #tmin = time[timeToUse].min()
  #tmax = time[timeToUse].max()
  total_win = int((tmax-tmin)*500)
  win_starts = range(0,total_win-win_len, overlap)

 # print(win_starts[-1])
 # print(np.sum(timeToUse))
#  data = data[:,:,timeToUse]


  #Run TGM
  preds,tgm,l_index,cv_membership,masks = hippo.models._lr_tgm(data, labels, cv, win_starts, win_len, doZscore,ddof,doAvg=doAvg)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)

  fnameStub = subDir + '/TGM_{sub}_{word}_{aorp}_win{win_len}_zscore{zscore}_avg{doAvg}_2F_LR_PDTW.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,word=word,aorp=actORpass,win_len=win_len,zscore=doZscore,doAvg=doAvg), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks,'labels':labels,'cv_membership':cv_membership})

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--analysis', required=True)
  parser.add_argument('--experiment', required=True)
  parser.add_argument('--subject', required=True)
  parser.add_argument('--word', required=True)
  parser.add_argument('--win_len',required=True,type=int)
  parser.add_argument('--actORpass',required=False)
  parser.add_argument('--doZscore',required=False,type=int)
  parser.add_argument('--ddof',required=False,type=int)
  parser.add_argument('--doAvg',required=False,type=int)

  args = parser.parse_args()

  analysis = args.analysis
  
  {
    'actORpass_krns2_lr_pdtw' : actORpass_krns2_lr_pdtw,
    'pooled_krns2_lr_pdtw' : pooled_krns2_lr_pdtw,
  }[analysis](args.experiment, args.subject,'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas', args.word, args.win_len,args.actORpass, args.doZscore, args.ddof,  args.doAvg)#,args.doPDTW)

# Old slugs: 
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'sss_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-5-150_notch-60-120_beatremoval-first_blinkremoval-first'


  #Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
 # tmin_a = wordOnset_a - 0.5
 # tmin_p = wordOnset_p - 0.5
 # if word == 'firstNoun':
 #   tmax_a = wordOnset_a + 4.5
 #   tmax_p = wordOnset_p + 4.5
 # elif word == 'verb':
 #   tmax_a = wordOnset_a + 4
 #   tmax_p = wordOnset_p + 4
 # else:
   # tmax_a = wordOnset_a + 3
  #  tmax_p = wordOnset_p + 3

 # if tmin_a < time.min():
 #   tmin_a = time.min()
 # if tmin_p < time.min():
 #   tmin_p = time.min()
 # if tmax_a > time.max():
 #   tmax_a = time.max()
 # if tmax_p > time.max():
  #  tmax_p = time.max()
  
 # timeToUse_a = np.squeeze(np.logical_and(time >= tmin_a,time < tmax_a))
#  timeToUse_p = np.squeeze(np.logical_and(time >= tmin_p,time < tmax_p))
