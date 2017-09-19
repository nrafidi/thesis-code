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


# Runs the TGM experiment, 2F CV, pooling active and passive sentences
def pooled_krns2_lr(experiment,subject, proc_slug,word,win_len,actORpass='pooled',doZscore=False,ddof=1,doAvg=False,do2Samp=False):
# Gets the stimulus IDs

  if word == 'firstNoun':
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['doctor', 'peach','dog','hammer','school','door','monkey','student']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
  elif word == 'verb':
    usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['found', 'kicked','inspected','touched']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
  else:
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['doctor.', 'peach.','dog.','hammer.','school.','door.','monkey.','student.']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions

  exp_sub = [(experiment, subject)]
  uels = hippo.query.get_uels_from_usis(usis.keys(), experiment_subjects=exp_sub)
  uels = {k : v for (k,v) in uels.iteritems() if len(v) > 0} #checking for empties
  id_uels = [(k, uels[k]) for k in uels.keys()] #putting uels in a list instead of a map (explicit ordering)
  print id_uels
  if len(id_uels[0][1]) == 0 :
    print 'nothing for %s %s' % (experiment, subject)
    exit()

  #Labels
  labels = [usis[k]['stimulus'] for k, _ in id_uels]

  #CV scheme
  usi_list = [k for k,_ in id_uels]
  cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
  cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))

  #2 Fold CV
  cv = np.ones((len(labels),), dtype=np.int);
  for l in list(set(labels)):
    print(l)
    indLabel = [i for i,x in enumerate(labels) if x == l]
    print(indLabel)
    cv[indLabel[0::2]] = 0

  #Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
  tmin = -0.5
  if word == 'firstNoun':
    tmax = 4.5
  elif word == 'verb':
    tmax = 4
  else:
    tmax = 3
  total_win = int((tmax-tmin)*500)
  overlap =  12
  win_starts = range(0,total_win-win_len, overlap)

  if not do2Samp:
  #List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
    evokeds = hippo.models.load_data(id_uels, tmin,tmax,proc_slug,baseline=None)
    data = np.array([e.data for e in evokeds])

  #Downsample data
    data = data[:,:,0::2]
  else:
    evokeds = hippo.models.load_data_noAvg(id_uels, tmin,tmax,proc_slug,baseline=None)
    data = np.array(evokeds)
  #Downsample and average data
    data_one = np.mean(data[:,range(0,15,2),:,0::2], axis=1)
    data_two = np.mean(data[:,range(1,15,2),:,0::2], axis=1)
    data = np.concatenate((data_one, data_two), axis=0)
    labels = labels + labels

  print(doAvg)
  #Run TGM
  preds,tgm,l_index,cv_membership,masks = hippo.models._lr_tgm(data, labels, cv, win_starts, win_len,doZscore,ddof,doAvg)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  fnameStub = subDir + '/TGM_{sub}_{word}_pooled_win{win_len}_zscore{doZ}_avg{doAvg}_2Samp{do2Samp}_2F_LR.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,word=word,win_len=win_len,doZ=doZscore,doAvg=doAvg,do2Samp=do2Samp), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks, 'cv_label':cv_label,'labels':labels,'cv_membership':cv_membership})

# Pooled
###################################################################################################################################

# Runs the TGM experiment, 2F CV, separately for active and passive sentences
def actORpass_krns2_lr(experiment,subject, proc_slug,word,win_len,actORpass,doZscore=True,ddof=1,doAvg=False,do2Samp=True):
  warnings.filterwarnings('error')
# Gets the stimulus IDs for all the first nouns
  if actORpass == 'active':
    if word == 'firstNoun':
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['doctor','dog','monkey','student']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
    elif word == 'verb':
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['found', 'kicked','inspected','touched']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None),('word_index_in_sentence', lambda wis : wis == 2)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
    else:
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['peach.','hammer.','school.','door.']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
  else:
    if word == 'firstNoun':
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['peach','hammer','school','door']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
    elif word == 'verb':
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['found', 'kicked','inspected','touched']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None),('word_index_in_sentence', lambda wis : wis == 3)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
    else:
      usis = hippo.query.query_usis([('stimuli_set',experiment), 
          ('stimulus',lambda s : s in ['doctor.','dog.','monkey.','student.']), # without periods, gets the first noun
          ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions

  exp_sub = [('krns2', subject)]
  uels = hippo.query.get_uels_from_usis(usis.keys(), experiment_subjects=exp_sub)
  uels = {k : v for (k,v) in uels.iteritems() if len(v) > 0} #checking for empties
  id_uels = [(k, uels[k]) for k in uels.keys()] #putting uels in a list instead of a map (explicit ordering)
  print id_uels
  if len(id_uels[0][1]) == 0 :
    print 'nothing for %s %s' % ('krns2', subject)
    exit()

  #Labels
  labels = [usis[k]['stimulus'] for k, _ in id_uels]
  labels = labels + labels #appends labels to itself
  usi_list = [k for k,_ in id_uels]

  #CV scheme
  usi_list = [k for k,_ in id_uels]
  cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
  cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))

  #2 Fold CV
  cv = np.ones((len(labels),), dtype=np.int);
  for l in list(set(labels)):
    print(l)
    indLabel = [i for i,x in enumerate(labels) if x == l]
    print(indLabel)
    cv[indLabel[0::2]] = 0

 #Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
  tmin = -0.5
  if word == 'firstNoun':
    tmax = 4.5
  elif word == 'verb':
    tmax = 4
  else:
    tmax = 3
  total_win = int((tmax-tmin)*500)
  overlap = 12;
  win_starts = range(0,total_win-win_len, overlap)

  #List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
  evokeds = hippo.models.load_data_noAvg(id_uels, tmin,tmax,proc_slug,baseline=None)
  data = np.array(evokeds)
  #Downsample and average data
  data_one = np.mean(data[:,range(0,15,2),:,0::2], axis=1)
  data_two = np.mean(data[:,range(1,15,2),:,0::2], axis=1)
  data = np.concatenate((data_one, data_two), axis=0)
#  scipy.io.savemat('/home/nrafidi/evokedDataTest.mat',mdict={'data':data,'cv':cv,'labels':labels})

  #Run TGM
  preds,tgm,l_index,cv_membership,masks = hippo.models._lr_tgm(data, labels, cv, win_starts, win_len,doZscore,ddof,doAvg=doAvg)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)

  fnameStub = subDir + '/TGM_{sub}_{word}_{aorp}_win{win_len}_zscore{zscore}_avg{doAvg}_2F_LR.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,word=word,aorp=actORpass,win_len=win_len,zscore=doZscore,doAvg=doAvg), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks,'cv_label':cv_label,'labels':labels,'cv_membership':cv_membership})

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
  parser.add_argument('--do2Samp',required=False,type=int)
#  parser.add_argument('--doPDTW',required=False,type=int)

  args = parser.parse_args()

  analysis = args.analysis
  
  {
    'actORpass_krns2_lr' : actORpass_krns2_lr,
    'pooled_krns2_lr' : pooled_krns2_lr,
  }[analysis](args.experiment, args.subject,'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas', args.word, args.win_len,args.actORpass, args.doZscore, args.ddof,  args.doAvg, args.do2Samp)#,args.doPDTW)

# Old slugs: 
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'sss_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-5-150_notch-60-120_beatremoval-first_blinkremoval-first'
