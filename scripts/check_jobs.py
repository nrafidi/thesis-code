import itertools
import os.path
import batch_experiments as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.SUBJECTS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.WIN_LENS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_AVGS,
                                   batch_exp.DO_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.REPS_TO_USES,
                                   batch_exp.RANDOM_STATES)
    job_id = 0
    successful_jobs = 0
    for grid in param_grid:
        exp = grid[0]
        sub = grid[1]
        sen = grid[2]
        word = grid[3]
        win_len = grid[4]
        overlap = grid[5]
        isPerm = grid[6]
        alg = grid[7]
        adj = grid[8]
        tm_avg = grid[9]
        tst_avg = grid[10]
        ni = grid[11]
        reps = grid[12]
        rs = grid[13]

        job_str = batch_exp.JOB_NAME.format(exp=exp,
                                            sub=sub,
                                            sen=sen,
                                            word=word,
                                            id=job_id)

        dir_str = batch_exp.JOB_DIR.format(exp=grid[0])

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)

        top_dir = TOP_DIR.format(exp=exp)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        save_dir = SAVE_DIR.format(top_dir=top_dir, sub=sub)
        fname = SAVE_FILE.format(dir=save_dir,
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
                                 rsP=rs)

        if os.path.isfile(fname + '.npz'):
            successful_jobs += 1

        if not os.path.isfile(err_str) or not os.path.isfile(out_str):
            # print('Job {} Did Not Run'.format(job_str))
            meow = 1
        else:
            if os.stat(err_str).st_size != 0:
                print('Job {} Failed'.format(job_str))
                with open(err_str, 'r') as fid:
                    print fid.read()
                print(grid)
            # else:
            #     with open(out_str, 'r') as fid:
            #         meow = fid.read()
            #         if 'Experiment parameters not valid.' in meow:
            #             print(meow)
            #             is_valid =  grid[5] <= grid[4]
            #             if grid[6] == 'coef':
            #                 is_valid = is_valid and (grid[9] == 2)
            #             if is_valid:
            #                 print(meow)
            #         else:
            #             successful_jobs += 1
            #             # print(job_id)

        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id))
