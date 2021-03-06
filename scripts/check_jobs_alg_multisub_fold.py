import itertools
import os.path
import batch_experiments_alg_multisub as batch_exp

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_alg_comp/'
SAVE_FILE = '{dir}TGM-alg-comp_multisub_pooled_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{fold}'


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
        rs = grid[8]
        word = grid[9]
        win_len = grid[10]
        fold = grid[11]

        if (win_len != 2 and ni != 1) or (not tm_avg and not tst_avg) or (not tm_avg and ni != 1):
            continue

        job_str = batch_exp.JOB_NAME.format(exp=exp,
                                            word=word,
                                            alg=alg,
                                            id=job_id)

        dir_str = batch_exp.JOB_DIR.format(exp=exp)

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)

        top_dir = TOP_DIR.format(exp=exp)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        fname = SAVE_FILE.format(dir=top_dir,
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

        if os.path.isfile(fname + '.npz'):
            successful_jobs += 1
            was_success = True
        else:
            was_success = False

        if not was_success:
            if not os.path.isfile(err_str) or not os.path.isfile(out_str):
                # print('Job {} never ran'.format(job_str))
                meow = 1
            else:
                # with open(out_str, 'r') as fid:
                #     meow = fid.read()
                #     if 'Skipping' in meow and 'already' not in meow:
                #         skipped = True
                #         if was_success:
                #             successful_jobs -= 1
                #         skipped_jobs += 1
                #     else:
                #         skipped=False
                if os.stat(err_str).st_size != 0: # and not skipped:
                    with open(err_str, 'r') as fid:
                        err_file = fid.read()
                        if not err_file.endswith('warnings.warn(_use_error_msg)\n'):
                            print('Job {} Failed'.format(job_str))
                            print(err_file)
                            print(grid)
        job_id += 1

    print('{}/{} jobs succeeded'.format(successful_jobs, job_id))
