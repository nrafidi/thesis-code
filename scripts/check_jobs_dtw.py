import itertools
import os.path
import batch_experiments_dtw as batch_exp

SAVE_FILE = '/share/volume0/nrafidi/DTW/EOS_dtw_sensor{i_sensor}_score_{exp}_{sub}_sen{sen0}_{radius}_{dist}_ni{ni}_{tmin}-{tmax}.npz'


if __name__ == '__main__':
    param_grid = itertools.product(batch_exp.EXPERIMENTS,
                                   batch_exp.SUBJECTS,
                                   batch_exp.RADIUS,
                                   batch_exp.TMINS,
                                   batch_exp.TMAXES,
                                   batch_exp.SEN0S,
                                   batch_exp.NINSTS,
                                   batch_exp.DISTS,
                                   batch_exp.SENSORS)
    job_id = 0
    successful_jobs = 0
    skipped_jobs = 0
    for grid in param_grid:
        exp = grid[0]
        sub = grid[1]
        radius = grid[2]
        tmin = grid[3]
        tmax = grid[4]
        sen0 = grid[5]
        ninst = grid[6]
        dist = grid[7]
        sensor = grid[8]

        job_str = batch_exp.JOB_NAME.format(sen0=sen0,
                                  radius=radius,
                                  ninst=ninst,
                                  id=job_id)

        dir_str = batch_exp.JOB_DIR.format(exp=exp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = batch_exp.ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = batch_exp.OUT_FILE.format(dir=dir_str, job_name=job_str)

        fname = SAVE_FILE.format(i_sensor=sensor,
                                 exp=exp,
                                 sub=sub,
                                 sen0=sen0,
                                 radius=radius,
                                 dist=dist,
                                 ni=ninst,
                                 tmin=tmin,
                                 tmax=tmax)

        if os.path.isfile(fname):
            successful_jobs += 1
            was_success = True
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
