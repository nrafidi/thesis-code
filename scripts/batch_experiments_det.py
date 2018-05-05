import itertools
import os.path
from subprocess import call, check_output
import time

MODES = ['acc'] #, 'coef']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] #['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']
SEN_TYPES = ['pooled'] #, 'active']
ANALYSES = ['det-type-first']
WIN_LENS = [12, 25, 50]
OVERLAPS = [12]
IS_PERMS = [False]  # True
ALGS = ['lr-l2']  # GNB
ADJS = ['zscore']
DO_TIME_AVGS = [False]
DO_TEST_AVGS = [True]#, True]  # True
NUM_INSTANCESS = [1, 2]
RANDOM_STATES = [1]

JOB_NAME = '{sub}-{analysis}-{id}'
JOB_DIR = '/share/volume0/nrafidi/krns2_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
                'subject={sub},sen_type={sen},analysis={analysis},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},mode={mode},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_det.sh'

    param_grid = itertools.product(MODES,
                                   OVERLAPS,
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
                                   SUBJECTS)
    job_id = 0
    for grid in param_grid:
        mode = grid[0]
        overlap = grid[1]
        isPerm = grid[2]
        alg = grid[3]
        adj = grid[4]
        tm_avg = grid[5]
        tst_avg = grid[6]
        ni = grid[7]
        rs = grid[8]
        win_len = grid[9]
        sen = grid[10]
        analysis = grid[11]
        sub = grid[12]

        job_str = JOB_NAME.format(sub=sub,
                                  analysis=analysis,
                                  id=job_id)

        dir_str = JOB_DIR
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    sub=sub,
                                    sen=sen,
                                    analysis=analysis,
                                    win_len=win_len,
                                    overlap=overlap,
                                    perm=isPerm,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    tst_avg=tst_avg,
                                    mode=mode,
                                    inst=ni,
                                    rs=rs,
                                    errfile=err_str,
                                    outfile=out_str)
        print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 150:
            time.sleep(30)
