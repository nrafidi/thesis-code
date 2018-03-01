import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mne
import argparse
import os
import numpy as np
import make_forward_solution as fwd_soln
import hippo.io
import hippo.query
from syntax_vs_semantics import load_data
from six import iteritems


RAW_FNAME = '/share/volume0/newmeg/{experiment}/data/{process_slug}/{subject}/{subject}_{experiment}_{block}_{process_slug}_raw.fif'
EMPTY_FNAME = '/share/volume0/newmeg/{experiment}/data/{process_slug}/{subject}/{subject}_{experiment}_EmptyRoom_{process_slug}_raw.fif'
INV_OP_PATH = '/share/volume0/newmeg/{experiment}/data/inv/{subject}'
INV_OP_FNAME = '{inv_path}/{subject}_{experiment}_{process_slug}_raw-{struct}-{spacing}-{loose}-{depth}-limit{limit}-rank{rank}-inv.fif'


def apply_inverse_operator(experiment,
                           subject,
                           process_slug,
                           spacing,
                           evoked, # Sensor data to localize (from hippocampus)
                           loose=0.2,
                           depth=0.8,
                           limit_depth_chs=True,
                           rank=None):
    struct = fwd_soln.sub_to_struct[experiment][subject]
    inv_path = INV_OP_PATH.format(experiment=experiment, subject=subject)

    inv_fname = INV_OP_FNAME.format(inv_path=inv_path,
                                    experiment=experiment,
                                    subject=subject,
                                    process_slug=process_slug,
                                    spacing=spacing,
                                    struct=struct,
                                    loose=loose,
                                    depth=depth,
                                    limit=limit_depth_chs,
                                    rank=rank)

    if not os.path.isfile(inv_fname):
        inv = make_inverse_operator(experiment,
                                    subject,
                                    process_slug,
                                    spacing,
                                    loose=loose,
                                    depth=depth,
                                    limit_depth_chs=limit_depth_chs,
                                    rank=rank)
    else:
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)

    src = mne.minimum_norm.apply_inverse(evoked, inv)

    return src


def extract_reg(experiment,
                subject,
                src,
                reg_label):
    structural_subject = fwd_soln.sub_to_struct[experiment][subject]

    labels = mne.read_labels_from_annot(structural_subject, parc='aparc', subjects_dir='/usr1/meg/structural',
                                        regexp=reg_label)

    # labels is a list (you can get more than one if you use a regular expression
    # each element is a mne.Label object. See http://martinos.org/mne/stable/generated/mne.Label.html#mne.Label
    # if you have more than one label you can add them together with the '+' operator. This will remove duplicates
    # therefore, if you use a regexp and get more than one label you will want to add them if you do
    # not want duplicate channels.

    # SourceEstimate.in_label returns a SourceEstimate object which only has the channels give to in_label
    # Search 'mne python api source estimate in label' for more information
    # the current link is http://martinos.org/mne/stable/generated/mne.SourceEstimate.html#mne.SourceEstimate.in_label
    src_only_label = src.in_label(labels[0])

    return src_only_label


def make_inverse_operator(experiment,
                          subject,
                          process_slug,
                          spacing,
                          loose=0.2,
                          depth=0.8,
                          limit_depth_chs=True,
                          rank=None):
  struct = fwd_soln.sub_to_struct[experiment][subject]
  empty_room_fname = EMPTY_FNAME.format(experiment=experiment,
                                        process_slug=process_slug,
                                        subject=subject)
  cov = mne.compute_raw_covariance(mne.io.Raw(empty_room_fname))

  fwd_path = fwd_soln.FWD_PATH.format(experiment=experiment,
                                      subject=subject)
  fwd_fname = fwd_soln.FWD_FNAME.format(fwd_path=fwd_path,
                                        subject=subject,
                                        experiment=experiment,
                                        process_slug=process_slug,
                                        struct=struct,
                                        spacing=spacing)
  if not os.path.isfile(fwd_fname):
      print('Forward solution has not been computed for {struct}'.format(struct=struct))
      print('Attempting to compute now')
      fwd = fwd_soln.make_forward_solution(experiment,
                                           subject,
                                           spacing,
                                           process_slug=process_slug)
  else:
      fwd = mne.read_forward_solution(fwd_fname)

  raw_fname = RAW_FNAME.format(experiment=experiment,
                               subject=subject,
                               process_slug=process_slug,
                               block='01')
  raw = mne.io.Raw(raw_fname)

  inv = mne.minimum_norm.make_inverse_operator(raw.info,
                                               fwd,
                                               cov,
                                               loose=loose,
                                               depth=depth,
                                               limit_depth_chs=limit_depth_chs,
                                               rank=rank)
  inv_path = INV_OP_PATH.format(experiment=experiment, subject=subject)
  if not os.path.exists(inv_path):
      try:
        os.mkdir(inv_path)
      except:
        os.mkdir(INV_OP_PATH.format(experiment=experiment, subject=''))
        os.mkdir(inv_path)

  inv_fname = INV_OP_FNAME.format(inv_path=inv_path,
                                  experiment=experiment,
                                  subject=subject,
                                  process_slug=process_slug,
                                  spacing=spacing,
                                  struct=struct,
                                  loose=loose,
                                  depth=depth,
                                  limit=limit_depth_chs,
                                  rank=rank)

  mne.minimum_norm.write_inverse_operator(inv_fname, inv)

  return inv


def is_in_long_sentence(ordered_sentence_usis):
    count = 0
    result = False
    for usi, annotation in ordered_sentence_usis:
        count += 1
        if count > 4:
            result = True
    for _ in range(count):
        yield result


def load_epochs(subject, experiment, filter_sets, tmin, tmax, proc=fwd_soln.DEFAULT_PROC):


    exp_sub = [(experiment, subject)]

    filter_set_usis = load_data.filtered_query(experiment, filter_sets)
    id_uels = list()
    uels = list()
    labels = list()
    indices_in_master_experiment_stimuli = list()
    set_num_uels = list()
    for set_usis in filter_set_usis:
        order_dict = dict([(kv[0], i) for i, kv in enumerate(set_usis)])
        set_uels = hippo.query.get_uels_from_usis([k for k, v in set_usis], experiment_subjects=exp_sub)
        set_uels = list(sorted(iteritems(set_uels), key=lambda id_uel: order_dict[id_uel[0]]))
        id_uels.extend(set_uels)
        num_uels = None
        for k, v in set_uels:
            uels.extend(v)
            if num_uels is None:
                num_uels = len(v)
            elif num_uels != len(v):
                raise ValueError('Unable to proceed, expected all usis to have the same number of uels')
        labels.append(np.array([load_data.punctuation_regex.sub('', a['stimulus']).lower() for _, a in set_usis]))
        indices_in_master_experiment_stimuli.append(
            np.array([a['index_in_master_experiment_stimuli'] for _, a in set_usis]))
        set_num_uels.append(num_uels)

    epochs = hippo.io.load_mne_epochs(uels, preprocessing=proc, baseline=None, tmin=tmin, tmax=tmax)

    labels = np.concatenate(labels, axis=0)
    indices_in_master_experiment_stimuli = np.concatenate(indices_in_master_experiment_stimuli, axis=0)

    time = np.arange(tmin, tmax + 1e-3, 1e-3)

    return epochs, labels, indices_in_master_experiment_stimuli, time


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment', required=True)
  parser.add_argument('--subject', required=True)
  parser.add_argument('--process_slug', default=fwd_soln.DEFAULT_PROC)
  parser.add_argument('--spacing', required=True, choices=[7, 20], type=int)
  parser.add_argument('--loose', type=float, default=0.2)
  parser.add_argument('--depth', type=float, default=0.8)
  parser.add_argument('--no_limit_depth_chs', action='store_false') # Doing this so it will default to true when flag not included
  parser.add_argument('--rank', default=None)
  args = parser.parse_args()

  # inv_op = make_inverse_operator(args.experiment,
  #                                args.subject,
  #                                args.process_slug,
  #                                args.spacing,
  #                                loose=args.loose,
  #                                depth=args.depth,
  #                                limit_depth_chs=args.no_limit_depth_chs,
  #                                rank=args.rank)

  epochs, labels, indices_in_master_experiment_stimuli, time = load_epochs(args.subject,
                                                                           args.experiment,
                                                                           filter_sets=[[load_data.is_in_active,
                                                                                         load_data.is_first_noun,
                                                                                         is_in_long_sentence]],
                                                                           tmin=-0.5,
                                                                           tmax=4.0,
                                                                           proc=args.process_slug)
  attrs = vars(epochs)
  for item in attrs.items():
      print(item)
  evokeds = mne.EvokedArray(np.ndarray(epochs), epochs.info, tmin=-0.5, comment='', nave=1, kind='average', verbose=None)
  src = apply_inverse_operator(args.experiment,
                               args.subject,
                               args.process_slug,
                               args.spacing,
                               evokeds,
                               loose=args.loose,
                               depth=args.depth,
                               limit_depth_chs=args.no_limit_depth_chs,
                               rank=args.rank)



  fig, ax = plt.subplots()
  h = ax.imshow(src.data, interpolation='nearest', aspect='auto')
  ax.set_xticks(range(0, len(time), 500))
  label_time = time[::500]
  label_time[np.abs(label_time) < 1e-15] = 0.0
  ax.set_xticklabels(label_time)
  ax.set_xlabel('Time')
  text_to_write = ['Det', 'Noun1', 'Verb', 'Det', 'Noun2.']
  max_line = 2.51 * 1000

  for i_v, v in enumerate(np.arange(0.0, max_line, 0.5 * 1000)):
      ax.axvline(x=v, color='r')
      if i_v < len(text_to_write):
          plt.text(v + 0.05 * 1000, 1500, text_to_write[i_v], color='r')

  plt.show()