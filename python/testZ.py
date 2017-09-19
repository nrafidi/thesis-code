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
# First Noun Pooled
# Runs the TGM experiment, LOO CV, pooling active and passive sentences
def firstNoun_krns2(experiment,stimuli_set, subject, proc_slug,win_len,actORpass):

# Gets the stimulus IDs for all the first nouns
  usis = hippo.query.query_usis([('stimuli_set','krns2'), 
      ('stimulus',lambda s : s in ['doctor', 'peach','dog','hammer','school','door','monkey','student']), # without periods, gets the first noun
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

  #CV scheme
  usi_list = [k for k,_ in id_uels]
  cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
  cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))
  cv = [cv_label_to_id[l] for l in cv_label] # Converts the CV labels to numbers 0-31

  #Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
  tmin = -0.5
  tmax = 4.5
  total_win = int((tmax-tmin)*500)
  overlap =  12
  win_starts = range(0,total_win-win_len, overlap)

  #List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
  evokeds = hippo.models.load_data(id_uels, tmin,tmax,proc_slug,baseline=None)
  data = np.array([e.data for e in evokeds])

  #Downsample data
  data = data[:,:,0::2]

  #Feature selection parameters
  feature_select='distance_of_means'
  feature_select_params={'number_of_features' : 500}

  #Run TGM
  preds, l_index,cv_membership,feature_masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  fnameStub = subDir + '/TGM_{sub}_firstNoun_pooled_win{win_len}_sss.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,win_len=win_len), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':feature_masks})

# First Noun Pooled
###################################################################################################################################
# Second Noun Pooled

# Runs the TGM experiment, LOO CV, pooling active and passive sentences
def secondNoun_krns2(experiment,stimuli_set, subject, proc_slug,win_len,actORpass):

# Gets the stimulus IDs for all the first nouns
  usis = hippo.query.query_usis([('stimuli_set','krns2'), 
      ('stimulus',lambda s : s in ['doctor.', 'peach.','dog.','hammer.','school.','door.','monkey.','student.']), # without periods, gets the first noun
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

  #CV scheme
  usi_list = [k for k,_ in id_uels]
  cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
  cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))
  cv = [cv_label_to_id[l] for l in cv_label] # Converts the CV labels to numbers 0-31

  #Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
  tmin = -0.5
  tmax = 4.5
  total_win = int((tmax-tmin)*500)
  overlap =  12
  win_starts = range(0,total_win-win_len, overlap)

  #List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
  evokeds = hippo.models.load_data(id_uels, tmin,tmax,proc_slug,baseline=None)
  data = np.array([e.data for e in evokeds])

  #Downsample data
  data = data[:,:,0::2]

  #Feature selection parameters
  feature_select='distance_of_means'
  feature_select_params={'number_of_features' : 500}

  #Run TGM
  preds, l_index,cv_membership,feature_masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  fnameStub = subDir + '/TGM_{sub}_secondNoun_pooled_win{win_len}.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,win_len=win_len), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks})

# Second Noun Pooled
###################################################################################################################################
# Verb Pooled

# Runs the TGM experiment, LOO CV, pooling active and passive sentences
def verb_krns2(experiment,stimuli_set, subject, proc_slug,win_len,actORpass):

# Gets the stimulus IDs for all the first nouns
  usis = hippo.query.query_usis([('stimuli_set','krns2'), 
      ('stimulus',lambda s : s in ['found', 'kicked','inspected','touched']), # without periods, gets the first noun
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

  #CV scheme
  usi_list = [k for k,_ in id_uels]
  cv_label = [':'.join(k.split(':')[0:2]) for k in usi_list] # removes the word-1 designation
  cv_label_to_id = dict(zip(list(set(cv_label)), xrange(len(set(cv_label)))))
  cv = [cv_label_to_id[l] for l in cv_label] # Converts the CV labels to numbers 0-31

  #Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
  tmin = -0.5
  tmax = 3.5
  total_win = int((tmax-tmin)*500)
  overlap =  1
  win_starts = range(0,total_win-win_len, overlap)

  #List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
  evokeds = hippo.models.load_data(id_uels, tmin,tmax,proc_slug,baseline=None)
  data = np.array([e.data for e in evokeds])

  #Downsample data
  data = data[:,:,0::2]

  #Feature selection parameters
  feature_select='distance_of_means'
  feature_select_params={'number_of_features' : 500}

  #Run TGM
  preds, l_index,cv_membership,feature_masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  fnameStub = subDir + '/TGM_{sub}_verb_pooled_win{win_len}.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,win_len=win_len), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks})

# Verb Pooled
###################################################################################################################################
# First Noun Separate

# Runs the TGM experiment, LOO CV, separately for active and passive sentences
def firstNoun_krns2_actORpass(experiment,stimuli_set, subject, proc_slug,win_len,actORpass):
  warnings.filterwarnings('error')
# Gets the stimulus IDs for all the first nouns
  if actORpass == 'active':
    usis = hippo.query.query_usis([('stimuli_set','krns2'), 
                                   ('stimulus',lambda s : s in ['doctor', 'dog','monkey','student']), # without periods, gets the first noun
                                   ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
  else:
    usis = hippo.query.query_usis([('stimuli_set','krns2'), 
                                   ('stimulus',lambda s : s in ['school', 'peach','hammer','door']), # without periods, gets the first noun
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
  tmax = 4.5
  total_win = int((tmax-tmin)*500)
  overlap = 1;
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
  feature_select_params={'number_of_features' : 500}

  #Run TGM
  preds, l_index,cv_membership,masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  if actORpass == 'active':
    fnameStub = subDir + '/TGM_{sub}_firstNoun_active_win{win_len}.mat'
  else:
    fnameStub = subDir + '/TGM_{sub}_firstNoun_passive_win{win_len}.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,win_len=win_len), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks})

# First Noun Separate
###################################################################################################################################
# Second Noun Separate

# Runs the TGM experiment, LOO CV, separately for active and passive sentences
def secondNoun_krns2_actORpass(experiment,stimuli_set, subject, proc_slug,win_len,actORpass):
  warnings.filterwarnings('error')
# Gets the stimulus IDs for all the first nouns
  if actORpass == 'passive':
    usis = hippo.query.query_usis([('stimuli_set','krns2'), 
                                   ('stimulus',lambda s : s in ['doctor.', 'dog.','monkey.','student.']), # without periods, gets the first noun
                                   ('sentence_id', lambda sid : sid != None)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
  else:
    usis = hippo.query.query_usis([('stimuli_set','krns2'), 
                                   ('stimulus',lambda s : s in ['school.', 'peach.','hammer.','door.']), # without periods, gets the first noun
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
  tmax = 4.5
  total_win = int((tmax-tmin)*500)
  overlap = 1;
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
  feature_select_params={'number_of_features' : 500}

  #Run TGM
  preds, l_index,cv_membership,masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  if actORpass == 'active':
    fnameStub = subDir + '/TGM_{sub}_secondNoun_active_win{win_len}.mat'
  else:
    fnameStub = subDir + '/TGM_{sub}_secondNoun_passive_win{win_len}.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,win_len=win_len), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks})

# Second Noun Separate
###################################################################################################################################
# Verb Separate

# Runs the TGM experiment, LOO CV, separately for  active and passive sentences
def verb_krns2_actORpass(experiment,stimuli_set, subject, proc_slug,win_len,actORpass):
  warnings.filterwarnings('error')
# Gets the stimulus IDs for all the first nouns
  if actORpass == 'active':
    usis = hippo.query.query_usis([('stimuli_set','krns2'), 
                                   ('stimulus',lambda s : s in ['found', 'kicked','inspected','touched']), # without periods, gets the first noun
                                   ('sentence_id', lambda sid : sid != None), ('word_index_in_sentence', lambda wis : wis == 2)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
  else:
    usis = hippo.query.query_usis([('stimuli_set','krns2'), 
                                   ('stimulus',lambda s : s in ['found', 'kicked','inspected','touched']), # without periods, gets the first noun
                                   ('sentence_id', lambda sid : sid != None), ('word_index_in_sentence', lambda wis : wis == 3)], include_annotations=['stimulus', 'sentence_id']) # excludes questions
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
  tmax = 4.5
  total_win = int((tmax-tmin)*500)
  overlap = 1;
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
  feature_select_params={'number_of_features' : 500}

  #Run TGM
  preds, l_index,cv_membership,masks = hippo.models._nb_tgm(data, labels, cv, win_starts, win_len, usi_list, feature_select,feature_select_params)
  tgm = hippo.models._tgm_from_preds(preds, labels, l_index, cv)

  #Save Directory
  saveDir = '/share/volume0/nrafidi/PA1_TGM'
  if not os.path.exists(saveDir):
    os.mkdir(saveDir)
  subDir = saveDir + '/' + subject
  if not os.path.exists(subDir):
    os.mkdir(subDir)
  if actORpass == 'active':
    fnameStub = subDir + '/TGM_{sub}_verb_active_win{win_len}.mat'
  else:
    fnameStub = subDir + '/TGM_{sub}_verb_passive_win{win_len}.mat'
  scipy.io.savemat(fnameStub.format(sub=subject,win_len=win_len), mdict={'preds':preds, 'tgm':tgm, 'feature_masks':masks})

# Verb Separate
###################################################################################################################################

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--analysis', required=True)
  parser.add_argument('--experiment', required=True)
  parser.add_argument('--stimuli_set', required=True)
  parser.add_argument('--subject', required=True)
  parser.add_argument('--win_len',required=True,type=int)
  parser.add_argument('--actORpass',required=False)

  args = parser.parse_args()

  analysis = args.analysis
  
  {
    'firstNoun_krns2' : firstNoun_krns2,
    'verb_krns2' : verb_krns2,
    'secondNoun_krns2' : secondNoun_krns2,
    'firstNoun_krns2_actORpass' : firstNoun_krns2_actORpass,
    'verb_krns2_actORpass' : verb_krns2_actORpass,
    'secondNoun_krns2_actORpass' : secondNoun_krns2_actORpass,
  }[analysis](args.experiment,args.stimuli_set, args.subject, 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas', args.win_len,args.actORpass)

# Old slugs: 
# 'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'
# 'sss_emptyroom-4-10-2-2_band-1-150_notch-60-120_beatremoval-first_blinkremoval-first'
# 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_band-5-150_notch-60-120_beatremoval-first_blinkremoval-first'
