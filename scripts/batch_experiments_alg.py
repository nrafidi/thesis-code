import itertools
import os.path
from subprocess import call, check_output
import time

EXPERIMENTS = ['krns2']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
WORDS = ['voice']
WIN_LENS = [2, 12, 25, 50, 100]
OVERLAPS = [2]
IS_PERMS = [False]
ALGS = ['lr-l1', 'lr-l2', 'svm-l1', 'svm-l2', 'gnb'],
ADJS = [None]
DO_TME_AVGS = [True, False]
DO_TST_AVGS = [True, False]
NUM_INSTANCESS = [1, 2, 5, 10]
RANDOM_STATES = [1]

JOB_NAME = '{exp}-{sub}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=144:00:00,mem=2GB -v ' \
                'experiment={exp},subject={sub},word={word},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_alg.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_TME_AVGS,
                                   DO_TST_AVGS,
                                   NUM_INSTANCESS,
                                   RANDOM_STATES,
                                   WORDS,
                                   WIN_LENS,
                                   SUBJECTS)
    job_id = 0
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
        sub = grid[11]

        job_str = JOB_NAME.format(exp=exp,
                                  sub=sub,
                                  word=word,
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=exp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    exp=exp,
                                    sub=sub,
                                    word=word,
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

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
