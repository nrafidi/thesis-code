import itertools
import os.path
import batch_experiments_rsa as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_RSA/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}RSA_{sub}_{word}_win{win_len}_ov{ov}_dist{dist}_avgTime{avgTm}_{draw}'



def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.DISTS,
                                   batch_exp.DO_TME_AVGS,
                                   batch_exp.WORDS,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SUBJECTS,
                                   batch_exp.DRAWS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        exp = grid[0]
        overlap = grid[1]
        dist = grid[2]
        tm_avg = grid[3]
        word = grid[4]
        win_len = grid[5]
        sub = grid[6]
        draw = grid[7]

        job_str = batch_exp.JOB_NAME.format(dist=dist,
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
                                 draw=draw,
                                 dist=dist,
                                 avgTm=bool_to_str(tm_avg))

        if os.path.isfile(fname + '.npz'):
            successful_jobs += 1
            was_success = True
            # print(grid)
        else:
            was_success = False

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
                    print('Job {} Failed'.format(job_str))
                    print(err_file)
                    print(grid)


        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id - skipped_jobs))
