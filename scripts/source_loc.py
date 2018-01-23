# source /home/python27/envs/preproc/bin/activate
# from http://martinos.org/mne/stable/auto_tutorials/plot_source_localization_basics.html
# and https://martinos.org/mne/stable/auto_tutorials/plot_forward.html#sphx-glr-auto-tutorials-plot-forward-py

import numpy as np
import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)
from mayavi import mlab  # noqa
from surfer import Brain  # noqa

mne.set_log_level('WARNING')

data_path = '/Users/alona/research/data/adjectives/dan_data/A/'
subj_dir = '/Users/alona/research/data/structural/a_test_real/'
subj='A'

#files required by this script
# MEG sensor data file - all relevant preprocessing should already be done
raw_fname = data_path + '/A_adjectives_05_trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_lp-150_notch-60-120_beatremoval-first_blinkremoval-first_raw.fif'
# trans file is made via coregistration "mne coreg"
fname_trans = data_path + '/A-trans.fif'

#Files that will be created with this script
#BEM files
bem_model_file = subj_dir + subj + '/bem/'+subj+'-5120-5120-5120-bem.fif'
bem_sol_file = subj_dir + subj + '/bem/'+subj+'-5120-5120-5120-bem-sol.fif'
#source space file
# Question: right now there seem to be waaaay too many sources (like 10x more than previously).
# Should figure out how to make there be fewer
src_file = subj_dir + subj + 'bem/A-oct6-src.fif'
#forward solution file
fwd_file = data_path + '/A-5120-5120-5120-bem-sol-oct6-fwd.fif'
#inverse solution file
inverse_file = data_path + '/A-5120-5120-5120-bem-sol-oct6-inv.fif'

# Need to make a evoked data file
# Question: does this mean we need an inv solution for every file???
raw = mne.io.Raw(raw_fname)
events = mne.find_events(raw, stim_channel='STI101')
e=events[(events[:,2]<16000),:] #button press
e=e[1:,:] # this is the instructions screen
e=e[range(0,len(e),2),:] #only keep the first word of every phrase

event_id = range(11,49)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)

epochs = mne.Epochs(raw, e, event_id, tmin, tmax)

# compute regularized noise covariance

noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'])

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)



evoked = epochs.average()
evoked.plot()
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='mag')

# boundary-element model (BEM)
model = mne.make_bem_model('A')
mne.write_bem_surfaces(bem_model_file, model)

bem_sol = mne.make_bem_solution(model)
mne.write_bem_solution(bem_sol_file, bem_sol)

# You may wish to check things here to make sure things look ok
mne.viz.plot_bem(subject=subj, subjects_dir=subj_dir,
                 brain_surfaces='white', orientation='coronal')


mne.viz.plot_alignment(raw.info, fname_trans, subject=subj, dig=True,
                       meg=['helmet', 'sensors'], subjects_dir=subj_dir)

# make forward solution
# Make dipole grid on surface, save to fif
mne.set_config('SUBJECTS_DIR',subj_dir)
src = mne.setup_source_space(subj, spacing='oct6')
mne.write_source_spaces(src_file, src)

#visualize the bem
mne.viz.plot_bem(subject=subj, subjects_dir=subj_dir,
                 brain_surfaces='white', src=src, orientation='coronal')

# visualize the sources
brain = Brain('A', 'lh', 'inflated', subjects_dir=subj_dir)
surf = brain.geo['lh']

vertidx = np.where(src[0]['inuse'])[0]

mlab.points3d(surf.x[vertidx], surf.y[vertidx],
              surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)



# Forward model

fwd = mne.make_forward_solution(raw.info, fname_trans, src, bem_sol_file)
mne.write_forward_solution(fwd_file, fwd)


# Restrict forward solution as necessary for MEG
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)

# make an inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)

write_inverse_operator(inverse_file,
                       inverse_operator)

# Compute inverse solution

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2,
                    method=method, pick_ori=None,verbose=1)


# visualize
_, peak_time = stc.get_peak(hemi='lh')
brain = stc.plot(initial_time=peak_time, surface='inflated', hemi='lh', subjects_dir=subj_dir)



