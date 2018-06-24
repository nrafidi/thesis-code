import itertools
import os.path
import batch_experiments_det_multisub_fold as batch_exp

TOP_DIR = '/share/volume0/nrafidi/krns2_TGM_LOSO_det/'
SAVE_FILE = '{dir}TGM-LOSO-det_multisub_{sen_type}_{analysis}_win{win_len}_ov{ov}_pr{perm}_' \
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
                                   batch_exp.DO_TIME_AVGS,
                                   batch_exp.DO_TEST_AVGS,
                                   batch_exp.NUM_INSTANCESS,
                                   batch_exp.RANDOM_STATES,
                                   batch_exp.WIN_LENS,
                                   batch_exp.SEN_TYPES,
                                   batch_exp.ANALYSES,
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
        sen = grid[9]
        analysis = grid[10]
        fold = grid[11]

        job_str = batch_exp.JOB_NAME.format(fold=fold,
                                  analysis=analysis,
                                  id=job_id)
        dir_str = batch_exp.JOB_DIR

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)


        fname = SAVE_FILE.format(dir=TOP_DIR,
                             sen_type=sen,
                             analysis=analysis,
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

        if os.path.isfile(fname + '.npz'):
            successful_jobs += 1
            was_success = True
        else:
            was_success = False

        if not os.path.isfile(err_str) or not os.path.isfile(out_str):
            # print('Job {} Did Not Run'.format(job_str))
            meow = 1
        else:
            if os.stat(err_str).st_size != 0 and (not was_success):
                with open(out_str, 'r') as fid:
                    out_info = fid.read()
                    print(out_info)
                with open(err_str, 'r') as fid:
                    err_file = fid.read()
                    print('Job {} Failed'.format(job_str))
                    print(err_file)
                    print(grid)

        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id - skipped_jobs))
