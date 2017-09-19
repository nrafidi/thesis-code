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

###################################################################################################################################
# Pooled
# Runs the TGM experiment, LOO CV, pooling active and passive sentences
def pooled_krns2(experiment,subject, proc_slug,word,win_len,actORpass,doZscore=True,ddof=1,numFeats=500,doPDTW=False):
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
  labels = labels + labels

  #CV scheme
  usi_list = [k for k,_ in id_uels]
  cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
  cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))
#  cv = [cv_label_to_id[l] for l in cv_label] # Converts the CV labels to numbers 0-31
  #LOO CV
  cv = range(0, 64);
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

  #List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
 # evokeds = hippo.models.load_data(id_uels, tmin,tmax,proc_slug,baseline=None)
 # data = np.array([e.data for e in evokeds])

  #Downsample data
#  data = data[:,:,0::2]

  evokeds = hippo.models.load_data_noAvg(id_uels, tmin,tmax,proc_slug,baseline=None)
  data = np.array(evokeds)
  #Downsample and average data
  data_one = np.mean(data[:,range(0,15,2),:,0::2], axis=1)
  data_two = np.mean(data[:,range(1,15,2),:,0::2], axis=1)
  data = np.concatenate((data_one, data_two), axis=0)
#  print(data.shape)

  #Feature selection parameters
  feature_select='distance_of_means'
  feature_select_params={'number_of_features' : numFeats}

  #Run TGM
  preds, l_index,cv_membership,feature_masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params,dozscore=doZscore,ddof=ddof)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  fnameStub = subDir + '/TGM_{sub}_{word}_pooled_win{win_len}_zscore{doZ}_numFeats{numFeats}.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,word=word,win_len=win_len,doZ=doZscore,numFeats=numFeats), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':feature_masks})

# Pooled
###################################################################################################################################
# Separate

# Runs the TGM experiment, LOO CV, separately for active and passive sentences
def actORpass_krns2(experiment,subject, proc_slug,word,win_len,actORpass,doZscore=True,ddof=1,numFeats=500,doPDTW=False):
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

  #LOO CV
  cv = range(0, 32);

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

  #Feature selection parameters
  feature_select='distance_of_means'
  feature_select_params={'number_of_features' : numFeats}

  #Run TGM
  preds, l_index,cv_membership,masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params,doZscore,ddof)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)

  fnameStub = subDir + '/TGM_{sub}_{word}_{aorp}_win{win_len}_zscore{zscore}_numFeats{numFeats}.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,word=word,aorp=actORpass,win_len=win_len,zscore=doZscore,numFeats=numFeats), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks})

#  Separate
###################################################################################################################################

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
  parser.add_argument('--numFeats',required=False,type=int)
  parser.add_argument('--doPDTW',required=False,type=int)

  args = parser.parse_args()

  analysis = args.analysis
  
  {
    'pooled_krns2' : pooled_krns2,
    'actORpass_krns2' : actORpass_krns2,
  }[analysis](args.experiment, args.subject,'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas', args.word, args.win_len,args.actORpass, args.doZscore, args.ddof, args.numFeats,args.doPDTW)

# Old slugs: 
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'sss_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-5-150_notch-60-120_beatremoval-first_blinkremoval-first'
