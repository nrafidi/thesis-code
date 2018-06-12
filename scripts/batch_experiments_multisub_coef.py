import itertools
import os.path
from subprocess import call, check_output
import time

EXPERIMENTS = ['krns2', 'PassAct3']
SEN_TYPES = ['active', 'passive']
WORDS = ['noun1', 'verb', 'noun2']
WIN_LENS = [50]
OVERLAPS = [5]
ALGS = ['lr-l2']  # GNB
ADJS = ['zscore']
DO_TIME_AVGS = [True]
NUM_INSTANCESS = [2]

JOB_NAME = '{exp}-coef-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

# JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'
JOB_Q_CHECK = 'expr $(qselect -q pool2 -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'

def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'

if __name__ == '__main__':

    # -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
    qsub_call = 'qsub -q pool2 -N {job_name} -l walltime=168:00:00,mem=16GB -v ' \
                'experiment={exp},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'adj={adj},alg={alg},doTimeAvg={tm_avg},' \
                'num_instances={inst},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_multisub_coef.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   OVERLAPS,
                                   ALGS,
                                   ADJS,
                                   DO_TIME_AVGS,
                                   NUM_INSTANCESS,
                                   WIN_LENS,
                                   SEN_TYPES,
                                   WORDS)
    job_id = 0
    for grid in param_grid:

        exp = grid[0]
        overlap = grid[1]
        alg = grid[2]
        adj = grid[3]
        tm_avg = grid[4]
        ni = grid[5]
        win_len = grid[6]
        sen = grid[7]
        word = grid[8]

        job_str = JOB_NAME.format(exp=exp,
                                  sen=sen,
                                  word=word,
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
                                    win_len=win_len,
                                    overlap=overlap,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    inst=ni,
                                    errfile=err_str,
                                    outfile=out_str)

        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
