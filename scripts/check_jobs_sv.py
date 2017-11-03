import itertools
import os.path
import batch_experiments_sv as batch_exp


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.SUBJECTS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.IS_PDTWS,
                                   batch_exp.IS_PERMS,
                                   batch_exp.NUM_FOLDSS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.REPS_TO_USES,
                                   batch_exp.RANDOM_STATES)
    job_id = 0
    successful_jobs = 0
    for grid in param_grid:
        job_str = batch_exp.JOB_NAME.format(exp=grid[0],
                                            sub=grid[1],
                                            sen=grid[2],
                                            word=grid[3],
                                            id=job_id)
        dir_str = batch_exp.JOB_DIR.format(exp=grid[0])

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)
        if os.path.isfile(err_str) and os.path.isfile(out_str):
            if os.stat(err_str).st_size != 0:
                print('Job {} Failed'.format(job_str))
                with open(err_str, 'r') as fid:
                    print fid.read()
                print(out_str)
                # with open(out_str, 'r') as fid:
                #     print fid.read()
            else:
                with open(out_str, 'r') as fid:
                    meow = fid.read()
                    if 'Experiment parameters not valid.' in meow:
                        if grid[6] <= grid[9]*16:
                            print(meow)
                    else:
                        successful_jobs += 1
                        print(job_id)


        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id))
