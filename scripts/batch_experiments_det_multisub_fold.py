import itertools
import os.path
from subprocess import call, check_output
import time

FOLDS = range(32)
SEN_TYPES = ['pooled'] #, 'active']
ANALYSES = ['a-dog', 'the-dog', 'det-type-first']
WIN_LENS = [50]
OVERLAPS = [5]
IS_PERMS = [False]  # True
ALGS = ['lr-l2']  # GNB
ADJS = ['zscore']
DO_TIME_AVGS = [True]
DO_TEST_AVGS = [True]#, True]  # True
NUM_INSTANCESS = [2]
RANDOM_STATES = [1]

JOB_NAME = '{fold}-{analysis}-{id}'
JOB_DIR = '/share/volume0/nrafidi/krns2_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
                'sen_type={sen},analysis={analysis},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},fold={fold},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_det_multisub_fold.sh'

    param_grid = itertools.product(OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_TIME_AVGS,
                                   DO_TEST_AVGS,
                                   NUM_INSTANCESS,
                                   RANDOM_STATES,
                                   WIN_LENS,
                                   SEN_TYPES,
                                   ANALYSES,
                                   FOLDS)
    job_id = 0
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

        job_str = JOB_NAME.format(fold=fold,
                                  analysis=analysis,
                                  id=job_id)

        dir_str = JOB_DIR
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    fold=fold,
                                    sen=sen,
                                    analysis=analysis,
                                    win_len=win_len,
                                    overlap=overlap,
                                    perm=isPerm,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    tst_avg=tst_avg,
                                    inst=ni,
                                    rs=rs,
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 300:
            time.sleep(30)
