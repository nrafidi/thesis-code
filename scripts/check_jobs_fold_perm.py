import itertools
import os.path
import batch_experiments_fold_perm as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_{mode}'

NEW_SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_{fold}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TIME_AVGS,
                                   batch_exp.DO_TEST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.REPS_TO_USES,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.SUBJECTS,
                                   batch_exp.FOLDS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        exp = grid[0]
        overlap = grid[1]
        isPerm = grid[2]
        alg = grid[3]
        adj = grid[4]
        tm_avg = grid[5]
        tst_avg = grid[6]
        ni = grid[7]
        reps = grid[8]
        rs = grid[9]
        win_len = grid[10]
        sen = grid[11]
        word = grid[12]
        sub = grid[13]
        fold = grid[14]

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
                                 rep=reps,
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
                                 rep=reps,
                                 rsP=rs,
                                 fold=fold)
        if os.path.isfile(old_job + '.npz'):
            skipped_jobs += 1
            was_success = True
        elif os.path.isfile(fname + '.npz'):
            successful_jobs += 1
            was_success = True
        else:
            was_success = False

        if not was_success:
            if not os.path.isfile(err_str) or not os.path.isfile(out_str):
                # print('Job {} Did Not Run'.format(job_str))
                meow = 1
            else:
                with open(out_str, 'r') as fid:
                    out_info = fid.read()
                if 'Experiment parameters not valid.' in out_info:
                    skipped_jobs += 1
                elif os.stat(err_str).st_size != 0:
                    with open(err_str, 'r') as fid:
                        err_file = fid.read()
                        if not err_file.endswith('warnings.warn(_use_error_msg)\n') and not ('Killed' in err_file):
                            if 'MemoryError' in err_file:
                                print('Job {} Failed'.format(job_str))
                            elif 'axis 0 with size 8' in err_file:
                                skipped_jobs += 1
                            else:
                                print(err_file)
                                print(grid)

        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id - skipped_jobs))
