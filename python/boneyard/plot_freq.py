import scipy.io
import argparse
import hippo.query
import hippo.TGM_models
import hippo.models
import cmumetadata.metaio as cmuio
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
import string
import os.path
import mne
from mne.time_frequency import tfr_morlet, psd_multitaper
from mne.datasets import somato


subject = 'C'

proc_slug =  'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'

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

#Time limits relative to word of interest (in SECONDS), window length and overlap (in SAMPLES)
tmin = -0.5
tmax = 4.5
#List of uels, tmin (s) [relative to word onset], tmax (s) [relative to word onset], preproc_slug, baseline correct?
evokeds = hippo.models.load_data_noAvg(id_uels, tmin,tmax,proc_slug,baseline=None)
epochs = mne.concatenate_epochs(evokeds)
epochs.resample(150.,npad='auto')
fig = epochs.plot_psd(fmin=2,fmax=140);
fig.savefig('psds_firstNoun_C.png');

fig = epochs.plot_psd_topomap(ch_type='grad',normalize=True);
fig.savefig('psds_topo_firstNoun_C.png');

#evokeds[0].resample(150., npad='auto')
#psds, freqs = psd_multitaper(evokeds[0], fmin=2, fmax=40, n_jobs=1)
#psds = 10 * np.log10(psds)
#scipy.io.savemat('psds_test.mat', mdict={'psds':psds})

#fig = evokeds[0].plot_psd(fmin=2,fmax=40)


#plt.show()
#evokeds[0].plot_psd_topomap(ch_type='grad', normalize=True)

#evokeds.plot_psd(fmin=2,fmax=40)
