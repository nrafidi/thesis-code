import itertools
import os.path
from subprocess import call, check_output
import time

def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


FOLDS = range(32)
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
EXPERIMENTS = ['krns2', 'PassAct3']
SEN_TYPES = ['active', 'passive', 'pooled']
WORDS = ['senlen', 'noun1', 'verb', 'voice', 'agent', 'patient', 'propid']
WIN_LENS = [50]
OVERLAPS = [5]
IS_PERMS = [False]
ALGS = ['lr-l2']
ADJS = ['zscore']
DO_TME_AVGS = [True]
DO_TST_AVGS = [True]
NUM_INSTANCESS = [2]
RANDOM_STATES = [1]

JOB_NAME = '{exp}-{sen}-{word}-{sub}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'
# JOB_Q_CHECK = 'expr $(qselect -q pool2 -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'

if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=192:00:00,mem=16GB -v ' \
                'experiment={exp},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'subject={sub},isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},fold={fold},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_eos_multisub_subcv_fold.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_TME_AVGS,
                                   DO_TST_AVGS,
                                   NUM_INSTANCESS,
                                   RANDOM_STATES,
                                   WIN_LENS,
                                   SEN_TYPES,
                                   WORDS,
                                   SUBJECTS,
                                    FOLDS)
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
        win_len = grid[9]
        sen = grid[10]
        word = grid[11]
        sub = grid[12]
        fold = grid[13]

        if fold > 15 and sen != 'pooled':
            continue

        if exp == 'krns2' and word == 'senlen':
            continue

        job_str = JOB_NAME.format(exp=exp,
                                  sen=sen,
                                  word=word,
                                  sub=sub,
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=exp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    exp=exp,
                                    sen=sen,
                                    word=word,
                                    sub=sub,
                                    win_len=win_len,
                                    overlap=overlap,
                                    perm=isPerm,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    tst_avg=tst_avg,
                                    fold=fold,
                                    inst=ni,
                                    rs=rs,
                                    errfile=err_str,
                                    outfile=out_str)


        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 300:
            time.sleep(30)
