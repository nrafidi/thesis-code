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


FOLDS = range(32)
EXPERIMENTS = ['PassAct3', 'krns2']
SEN_TYPES = ['active', 'passive', 'pooled']
WORDS = ['senlen', 'noun1', 'verb', 'voice', 'agent', 'patient', 'propid']
WIN_LENS = [50]
OVERLAPS = [5]
IS_PERMS = [True]
ALGS = ['lr-l2']
ADJS = ['zscore']
DO_TME_AVGS = [True]
DO_TST_AVGS = [True]
NUM_INSTANCESS = [2]
RANDOM_STATES = range(100)

JOB_NAME = '{exp}-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'
# JOB_Q_CHECK = 'expr $(qselect -q pool2 -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'

if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=192:00:00,mem=16GB -v ' \
                'experiment={exp},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},fold={fold},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_eos_multisub_fold.sh'

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

        if fold > 15 and sen != 'pooled':
            continue

        if exp == 'krns2' and word == 'senlen':
            continue

        if word in ['propid', 'voice', 'senlen', 'noun1'] and sen != 'pooled':
            continue

        if exp == 'PassAct3' and word in ['agent', 'patient', 'propid'] and fold > 7:
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

        top_dir = TOP_DIR.format(exp=exp)
        fname = SAVE_FILE.format(dir=top_dir,
                                 sen_type=sen,
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
                                 mode='acc')

        if not os.path.isfile(fname + '.npz'):
            call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
