import argparse
import os
from subprocess import call
import fnmatch


sub_to_raw_dir = {
  'PassAct3' : {
      'B' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/004',
      'P' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/059',
      'R' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/061',
      'T' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/063',
      'U' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/064',
      'V' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/065'
  },
    'PassAct3_Aud' : {
        'B' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/037',
        'E' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/046',
        'A' : '/bigbrain/bigbrain.usr1/meg/structural_raw_all/053'
    }
}

MRICRON = '/bigbrain/bigbrain.usr1/apps/mricron/dcm2nii -g n {dir}/*.dcm'
FREE_SURFER = 'recon-all -i {nifti} -subjid {subjid} | tee "{subjid}"_surface.log'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=sub_to_raw_dir.keys())
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    experiment = args.experiment
    subject = args.subject

    if subject not in sub_to_raw_dir[experiment].keys():
        raise AssertionError('Subject structural folder unknown.')

    subjid = experiment + '_' + subject

    root_dir = sub_to_raw_dir[experiment][subject]

    dir_to_use = ''
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        if dir_name[-2:] == '/3':
            if len(file_list) != 176:
                raise AssertionError('Directory {} has unexpected number of dicoms: {}'.format(dir_name, len(file_list)))
            dir_to_use = dir_name
            break

    call(MRICRON.format(dir=dir_to_use), shell=True)

    nifti = [fn for fn in os.listdir(dir_to_use) if fnmatch.fnmatch(fn, 'co*.nii')]

    if len(nifti) > 1:
        raise AssertionError('There seems to be more than one nifti file')

    nifti = dir_to_use + '/' + nifti[0]

    free_surfer_call = FREE_SURFER.format(nifti=nifti, subjid=subjid)

    print('Run the following in big-brain:\n{free}'.format(free=free_surfer_call))





