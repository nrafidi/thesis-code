import itertools
import os.path
import batch_experiments_sv as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_OH/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}OH_{sub}_{sen_type}_{word}_pr{perm}_' \
            'F{num_folds}_alg{alg}_adjX-{adjX}_adjY-{adjY}_avgTest{avgT}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_rsCV{rsC}'

CV_RAND_STATE = 12191989

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
                                   batch_exp.IS_PERMS,
                                   batch_exp.NUM_FOLDSS,
                                   batch_exp.ALGS,
                                   batch_exp.ADJS,
                                   batch_exp.ADJS,
                                   batch_exp.TST_AVGS,
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
        perm = grid[4]
        nf = grid[5]
        alg = grid[6]
        adjX = grid[7]
        adjY = grid[8]
        tst_avg = grid[9]
        inst = grid[10]
        rep = grid[11]
        rs = grid[12]

        total_jobs += 1

        top_dir = TOP_DIR.format(exp=exp)
        save_dir = SAVE_DIR.format(top_dir=top_dir, sub=sub)

        fname = SAVE_FILE.format(dir=save_dir,
                                 sub=sub,
                                 sen_type=sen,
                                 word=word,
                                 perm=bool_to_str(perm),
                                 num_folds=nf,
                                 alg=alg,
                                 adjX=adjX,
                                 adjY=adjY,
                                 avgT=bool_to_str(tst_avg),
                                 inst=inst,
                                 rep=rep,
                                 rsP=rs,
                                 rsC=CV_RAND_STATE)

        if os.path.isfile(fname + '.npz'):
            successful_jobs += 1

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

                with open(err_str, 'r') as fid:
                    meow = fid.read()
                    if 'MemoryError' in meow:
                        print('Job {} Failed'.format(job_str))
                        print('MemoryError')
                        print(out_str)
                        print(grid)
                    elif 'Killed' in meow:
                        print('Job {} Failed'.format(job_str))
                        print('Killed')
                        print(out_str)
                        print(grid)
                    elif 'False' in meow:
                        print('Weird error')
                    else:
                        print(meow)
            with open(out_str, 'r') as fid:
                meow = fid.read()
                if 'Skipping' in meow:
                    total_jobs -= 1

        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, total_jobs))
