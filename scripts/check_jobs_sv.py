import itertools
import os.path
import batch_experiments_sv as batch_exp


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.SUBJECTS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.WORDS,
                                   batch_exp.MODELS,
                                   batch_exp.INC_ART1,
                                   batch_exp.INC_ART2,
                                   batch_exp.ONLY_ART1,
                                   batch_exp.ONLY_ART2,
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
    total_jobs = 0
    for grid in param_grid:
        exp = grid[0]
        sub = grid[1]
        sen = grid[2]
        word = grid[3]
        model = grid[4]
        art1 = grid[5]
        art2 = grid[6]
        oart1 = grid[7]
        oart2 = grid[8]
        pdtw = grid[9]
        perm = grid[10]
        nf = grid[11]
        alg = grid[12]
        adj = grid[13]
        inst = grid[14]
        rep = grid[15]
        rs = grid[16]

        job_str = batch_exp.JOB_NAME.format(exp=exp,
                                              sub=sub,
                                              sen=sen,
                                              word=word,
                                              id=job_id)
        dir_str = batch_exp.JOB_DIR.format(exp=exp)

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)
        if os.path.isfile(err_str) and os.path.isfile(out_str):
            if os.stat(err_str).st_size != 0:
                print('Job {} Failed'.format(job_str))
                with open(err_str, 'r') as fid:
                    meow = fid.read()
                    if 'MemoryError' in meow:
                        print('MemoryError')
                    elif 'Killed' in meow:
                        print('Killed')
                    else:
                        print(meow)
                print(out_str)
                print(grid)
                total_jobs += 1
                # with open(out_str, 'r') as fid:
                #     print fid.read()
            else:
                with open(out_str, 'r') as fid:
                    meow = fid.read()
                    if 'Experiment parameters not valid.' not in meow:
                        successful_jobs += 1
                        total_jobs += 1
                        print('{} succeeded with parameters: {}'.format(job_id, grid))


        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, total_jobs))
