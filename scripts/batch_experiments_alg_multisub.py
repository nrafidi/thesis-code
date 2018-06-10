import itertools
import os.path
from subprocess import call, check_output
import time

EXPERIMENTS = ['krns2']
WORDS = ['voice', 'verb']
WIN_LENS = [2]#, 12, 25, 50, 100, 150]
OVERLAPS = [2]
IS_PERMS = [False]
ALGS = ['lr-None', 'gnb-None', 'lr-l2', 'lr-l1', 'svm-l1', 'svm-l2', 'gnb']
ADJS = ['zscore'] #, None]
DO_TME_AVGS = [True] #, False]
DO_TST_AVGS = [True] #, False]
NUM_INSTANCESS = [1]#, 2, 5, 10]
RANDOM_STATES = [1]
FOLDS = range(32)

JOB_NAME = '{exp}-{word}-{alg}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=144:00:00,mem=8GB -v ' \
                'experiment={exp},word={word},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force={force} ' \
                '-e {errfile} -o {outfile} submit_experiment_alg_multisub_fold.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   OVERLAPS,
                                   IS_PERMS,
                                   ADJS,
                                   DO_TME_AVGS,
                                   DO_TST_AVGS,
                                   NUM_INSTANCESS,
                                   RANDOM_STATES,
                                   WORDS,
                                   WIN_LENS,
                                   ALGS,
                                   FOLDS)
    job_id = 0
    for grid in param_grid:
        exp = grid[0]
        overlap = grid[1]
        isPerm = grid[2]
        adj = grid[3]
        tm_avg = grid[4]
        tst_avg = grid[5]
        ni = grid[6]
        rs = grid[7]
        word = grid[8]
        win_len = grid[9]
        alg = grid[10]
        fold = grid[11]
        force = False

        # if win_len != 2 and ni != 1:
        #     continue
        # if not tm_avg and not tst_avg:
        #     continue

        job_str = JOB_NAME.format(exp=exp,
                                  word=word,
                                  alg=alg,
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=exp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    exp=exp,
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
                                    force=force,
                                    errfile=err_str,
                                    outfile=out_str)
        print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 200:
            time.sleep(30)
