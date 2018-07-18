import itertools
import os.path
from subprocess import call, check_output
import time

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO_EOS/'
SAVE_FILE = '{dir}TGM-LOSO-EOS_multisub_{sen_type}_{word}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'


def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


FOLDS = range(8)
EXPERIMENTS = ['krns2']
SEN_TYPES = ['pooled']
WORDS = ['verb', 'voice', 'propid']
WIN_LENS = [50]
OVERLAPS = [5]
IS_PERMS = [False]
ALGS = ['lr-l2']
ADJS = ['zscore']
DO_TME_AVGS = [True]
DO_TST_AVGS = [True]
NUM_INSTANCESS = [2]
RANDOM_STATES = [1]
CV_RANDOM_STATES = range(100)
NUM_FOLDS = [2, 4, 8]

JOB_NAME = '{exp}-kf-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'
# JOB_Q_CHECK = 'expr $(qselect -q pool2 -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'

if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=192:00:00,mem=32GB -v ' \
                'experiment={exp},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'num_folds={nf},isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},fold={fold},' \
                'cv_random_state={cv_rs},doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=True, ' \
                '-e {errfile} -o {outfile} submit_experiment_eos_kf_multisub_fold.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_TME_AVGS,
                                   DO_TST_AVGS,
                                   NUM_INSTANCESS,
                                   RANDOM_STATES,
                                   CV_RANDOM_STATES,
                                   WIN_LENS,
                                   SEN_TYPES,
                                   WORDS,
                                   NUM_FOLDS,
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
        cv_rs = grid[9]
        win_len = grid[10]
        sen = grid[11]
        word = grid[12]
        num_folds = grid[13]
        fold = grid[14]

        if fold >= num_folds:
            continue

        if exp == 'krns2' and word == 'senlen':
            continue

        if word in ['propid', 'voice', 'senlen', 'noun1'] and sen != 'pooled':
            continue

        if num_folds > 2 and word == 'propid':
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
                                    nf=num_folds,
                                    perm=isPerm,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    tst_avg=tst_avg,
                                    fold=fold,
                                    inst=ni,
                                    rs=rs,
                                    cv_rs=cv_rs,
                                    errfile=err_str,
                                    outfile=out_str)

        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 75:
            time.sleep(30)
