import itertools
import os.path
import batch_experiments_alg_l2 as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_alg_comp/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-alg-comp_{sub}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}'


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
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.DO_TST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WORDS,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SUBJECTS)
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
        rs = grid[8]
        word = grid[9]
        win_len = grid[10]
        sub = grid[11]


        job_str = batch_exp.JOB_NAME.format(exp=exp,
                                            sub=sub,
                                            word=word,
                                            id=job_id)

        dir_str = batch_exp.JOB_DIR.format(exp=exp)

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)

        top_dir = TOP_DIR.format(exp=exp)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        save_dir = SAVE_DIR.format(top_dir=top_dir, sub=sub)
        fname = SAVE_FILE.format(dir=save_dir,
                                 sub=sub,
                                 word=word,
                                 win_len=win_len,
                                 ov=overlap,
                                 perm=bool_to_str(isPerm),
                                 alg=alg,
                                 adj=adj,
                                 avgTm=bool_to_str(tm_avg),
                                 avgTst=bool_to_str(tst_avg),
                                 inst=ni,
                                 rsP=rs)
        # if job_id in [781, 794, 807, 820, 833]:
        #     print(grid)
        if os.path.isfile(fname + '.npz'):
            successful_jobs += 1
            was_success = True
            # print(grid)
        else:
            was_success = False

        if win_len != 2 and ni != 1:
            skipped_jobs += 1
            if was_success:
                successful_jobs -= 1
        elif not tm_avg and not tst_avg:
            skipped_jobs += 1
            if was_success:
                successful_jobs -= 1
        else:
            if not os.path.isfile(err_str) or not os.path.isfile(out_str):
                # print('Job {} Did Not Run'.format(job_str))
                meow = 1
            else:
                with open(out_str, 'r') as fid:
                    meow = fid.read()
                    if 'Skipping' in meow and 'already' not in meow:
                        skipped = True
                        if was_success:
                            successful_jobs -= 1
                        skipped_jobs += 1
                    else:
                        skipped=False
                if os.stat(err_str).st_size != 0 and (not was_success) and not skipped:
                    with open(err_str, 'r') as fid:
                        err_file = fid.read()
                        # print('Job {} Failed'.format(job_str))
                        # print(err_file)
                        # print(grid)


        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id - skipped_jobs))
