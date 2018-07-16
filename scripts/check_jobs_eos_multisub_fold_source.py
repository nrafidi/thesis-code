import itertools
import os.path
import batch_experiments_eos_multisub_fold_source as batch_exp

TOP_DIR = '/share/volume0/nrafidi/PassAct3_TGM_LOSO_EOS_SOURCE/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_{reg}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.DO_TST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.HEMIS,
                                   batch_exp.REGIONS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                    batch_exp.FOLDS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        overlap = grid[0]
        isPerm = grid[1]
        alg = grid[2]
        adj = grid[3]
        tm_avg = grid[4]
        tst_avg = grid[5]
        ni = grid[6]
        rs = grid[7]
        win_len = grid[8]
        hemi = grid[9]
        reg = grid[10]
        sen = grid[10]
        word = grid[11]
        fold = grid[12]

        if fold > 15 and sen != 'pooled':
            continue

        if word in ['propid', 'voice', 'senlen', 'noun1'] and sen != 'pooled':
            continue

        if word in ['agent', 'patient', 'propid']:
            if sen != 'pooled' and fold > 7:
                continue
            elif sen == 'pooled' and fold > 15:
                continue

        job_str = batch_exp.JOB_NAME.format(sen=sen,
                                            word=word,
                                            id=job_id)

        err_str = batch_exp.ERR_FILE.format(dir=batch_exp.JOB_DIR, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=batch_exp.JOB_DIR, job_name=job_str)

        top_dir = TOP_DIR
        old_job = SAVE_FILE.format(dir=top_dir,
                                 sen_type=sen,
                                 word=word,
                                 win_len=win_len,
                                 ov=overlap,
                                 perm=bool_to_str(isPerm),
                                 alg=alg,
                                 adj=adj,
                                 avgTm=bool_to_str(tm_avg),
                                 avgTst=bool_to_str(tst_avg),
                                 inst=ni,
                                 rsP=rs,
                                   reg='{}-{}'.format(reg, hemi),
                                 fold=fold)

        if os.path.isfile(old_job + '.npz'):
            successful_jobs += 1
            was_success = True
        else:
            was_success = False

        if not was_success:
            if not os.path.isfile(err_str) or not os.path.isfile(out_str):
                # print('Job {} Did Not Run'.format(job_str))
                meow = 1
            else:
                if os.stat(err_str).st_size != 0:
                    with open(err_str, 'r') as fid:
                        err_file = fid.read()
                        if not err_file.endswith('warnings.warn(_use_error_msg)\n'):
                            too_long = 'exceeded limit' in err_file
                            zsl = 'ValueError: Class label' in err_file
                            if not too_long and not zsl:
                                print('Job {} Failed'.format(job_str))
                                print err_file
                            elif zsl:
                                skipped_jobs += 1
                            else:
                                print('Job {} Overtime'.format(job_str))

        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id - skipped_jobs))
