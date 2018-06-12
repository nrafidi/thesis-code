import itertools
import os.path
import batch_experiments_multisub_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_FILE = '{dir}coef-TGM_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_ni{inst}_coef'

def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.OVERLAPS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.DO_TIME_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        exp = grid[0]
        overlap = grid[1]
        alg = grid[2]
        adj = grid[3]
        tm_avg = grid[4]
        ni = grid[5]
        win_len = grid[6]
        sen = grid[7]
        word = grid[8]

        job_str = batch_exp.JOB_NAME.format(exp=exp,
                                            sen=sen,
                                            word=word,
                                            id=job_id)

        dir_str = batch_exp.JOB_DIR.format(exp=exp)

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)

        top_dir = TOP_DIR.format(exp=exp)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        fname = SAVE_FILE.format(dir=top_dir,
                                 sen_type=sen,
                                 word=word,
                                 win_len=win_len,
                                 ov=overlap,
                                 alg=alg,
                                 adj=adj,
                                 avgTm=bool_to_str(tm_avg),
                                 inst=ni)
        if os.path.isfile(fname + '.npz'):
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
                        print(err_file)
                        print(grid)
                        # if not err_file.endswith('warnings.warn(_use_error_msg)\n') and not ('Killed' in err_file):
                        #     if 'MemoryError' in err_file:
                        #         print('Job {} Failed Memory Error'.format(job_str))
                        #     elif 'IndexError' in err_file and word=='noun2' and exp == 'PassAct3':
                        #         skipped_jobs += 1
                        #     else:
                        #         print(err_file)
                        #         print(grid)

        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id - skipped_jobs))
