import itertools
import os.path
from subprocess import call, check_output
import time

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO_{sub}_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'nr{rep}_rsPerm{rsP}_{mode}'

FOLDS = range(16) #, 'coef']
EXPERIMENTS = ['krns2', 'PassAct3']
SEN_TYPES = ['active', 'passive']
WORDS = ['noun1', 'verb', 'noun2']
WIN_LENS = [50]
OVERLAPS = [5]
IS_PERMS = [True]  # True
ALGS = ['lr-l2']  # GNB
ADJS = ['zscore']
DO_TIME_AVGS = [True]
DO_TEST_AVGS = [True]
NUM_INSTANCESS = [2]
RANDOM_STATES = range(100)

JOB_NAME = '{exp}-dur-{sen}-{word}-{id}'
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
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},fold={fold},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_multisub_fold.sh'

    param_grid = itertools.product(EXPERIMENTS,
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
                                   WORDS,
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
        fold = grid[12]

        if exp == 'PassAct3' and fold > 7 and word == 'noun2':
            continue

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

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
