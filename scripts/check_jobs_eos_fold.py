import itertools
import os.path
import batch_experiments_eos_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'
NEW_SAVE_FILE = '{dir}TGM-LOSO-EOS_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.FOLDS,
                                   batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.DO_TST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.SUBJECTS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        fold = grid[0]
        exp = grid[1]
        overlap = grid[2]
        isPerm = grid[3]
        alg = grid[4]
        adj = grid[5]
        tm_avg = grid[6]
        tst_avg = grid[7]
        ni = grid[8]
        rs = grid[9]
        win_len = grid[10]
        sen = grid[11]
        word = grid[12]
        sub = grid[13]


        job_str = batch_exp.JOB_NAME.format(exp=exp,
                                            sub=sub,
                                            sen=sen,
                                            word=word,
                                            id=job_id)

        dir_str = batch_exp.JOB_DIR.format(exp=exp)

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)

        top_dir = TOP_DIR.format(exp=exp)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        save_dir = SAVE_DIR.format(top_dir=top_dir, sub=sub)
        old_job = SAVE_FILE.format(dir=save_dir,
                                 sub=sub,
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
                                 mode='acc')
        fname = NEW_SAVE_FILE.format(dir=save_dir,
                                 sub=sub,
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
                                 fold=fold)

        # if os.path.isfile(old_job + '.npz'):
        #     skipped_jobs += 1
        #     was_success = True
        if os.path.isfile(fname + '.npz'):
            successful_jobs += 1
            was_success = True
        else:
            was_success = False

        if exp == 'PassAct3' and word in ['agent', 'patient', 'propid'] and fold > 15:
            skipped_jobs +=1
        elif not was_success:
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
