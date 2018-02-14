import mne
import argparse
from os.path import join as pjoin
import os
import fnmatch


sub_to_struct = {
  '20questions' : {
    'A' : 'MP',
    'B' : 'DP',
    'C' : 'C',
    'D' : 'DB',
    'E' : 'JW',
    'F' : 'AD',
    'G' : 'JM',
    'H' : 'TS',
    'I' : 'RV'
  },
  'PassAct3' : {
      'B' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/004',
      'G' : 'krns5A', # Potentially mixed up with subject A
      'C' : 'struct1',
      'D' : 'krns5B',
      'E' : 'struct2',
      'P' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/059',
      'R' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/061',
      'T' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/063',
      'U' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/064',
      'V' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/065'
  },
    'PassAct3_Aud' : {
        'C' : 'krns5D',
        'D' : 'struct2',
        'B' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/037',
        'E' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/046',
        'A' : 'NA', #'/bigbrain/bigbrain.usr1/meg/structural_raw_all/053'
    }
}


DEFAULT_PROC = 'trans-D_nsb-5_cb-0_emptyroom-4-10-2-2_lp-150_notch-60-120_beatremoval-first_blinkremoval-first'

TRANS_FNAME = '/share/volume0/newmeg/{experiment}/data/raw/{subject}/{subject}_{experiment}_01_raw-trans-{struct}-1.fif'
EMPTY_FNAME = '/share/volume0/newmeg/{experiment}/data/{process_slug}/{subject}/{subject}_{experiment}_EmptyRoom_{process_slug}_raw.fif'
FWD_PATH = '/share/volume0/newmeg/{experiment}/data/fwd/{subject}'
FWD_FNAME = '{fwd_path}/{subject}_{experiment}_{process_slug}_raw-{struct}-{spacing}-fwd.fif'
PROC_FNAME = '/share/volume0/newmeg/{experiment}/data/{process_slug}/{subject}/{subject}_{experiment}_01_{process_slug}_raw.fif'
BEM_PATH = '/bigbrain/bigbrain.usr1/meg/structural/{struct}/bem/'
SRC_FNAME = '/bigbrain/bigbrain.usr1/meg/structural/{struct}/bem/{struct}-{spacing}-src.fif'
SUBJ_DIR = '/bigbrain/bigbrain.usr1/meg/structural/'


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment", required=True)
  parser.add_argument("--subject", required=True)
  parser.add_argument("--process_slug", default=DEFAULT_PROC)
  parser.add_argument('--spacing', type=int, required=True)
  args = parser.parse_args()

  experiment = args.experiment
  subject = args.subject
  process_slug = args.process_slug
  spacing = args.spacing

  struct = sub_to_struct[experiment][subject]
  if struct == 'NA':
      raise IOError('Freesurfer Reconstruction has not yet been done for this subject.')

  trans_fname = TRANS_FNAME.format(experiment=experiment, subject=subject, struct=struct)

  if not os.path.isfile(trans_fname):
      raise IOError('Coregistration has not yet been done for this subject')

  trans = mne.read_trans(trans_fname)

  # Needed for Inverse Solution
  # empty_room_fname = EMPTY_FNAME.format(experiment=experiment, subject=subject, process_slug=process_slug)
  # cov = mne.compute_raw_covariance(mne.io.Raw(empty_room_fname))

  fwd_path = FWD_PATH.format(experiment=experiment, subject=subject)
  if not os.path.exists(fwd_path):
      os.mkdir(fwd_path)

  fwd_fname = FWD_FNAME.format(fwd_path=fwd_path, subject=subject, experiment=experiment, process_slug=process_slug,
                               struct=struct, spacing=spacing)

  raw = mne.io.Raw(PROC_FNAME.format(experiment=experiment, subject=subject, process_slug=process_slug))

  bem_path = [fn for fn in os.listdir(BEM_PATH.format(struct=struct)) if fnmatch.fnmatch(fn, '*-bem-sol.fif')]

  if len(bem_path) == 0:
      raise IOError('BEM has not yet been done for this subject')

  bem_fname = pjoin(BEM_PATH.format(struct=struct), bem_path[0])

  src_file = SRC_FNAME.format(struct=struct, spacing=spacing)

  mne.set_config('SUBJECTS_DIR', SUBJ_DIR)

  # Not sure how to make sure this runs effectively
  src = mne.setup_source_space(struct, spacing='oct6')
  mne.write_source_spaces(src_file, src)


  fwd = mne.make_forward_solution(raw.info, trans,
                                  src=src,
                                  bem=bem_fname)

  mne.write_forward_solution(fwd_path, fwd)