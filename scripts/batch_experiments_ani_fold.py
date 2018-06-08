import itertools
import os.path
from subprocess import call, check_output
import time

TOP_DIR = '/share/volume0/nrafidi/{exp}_TGM_LOSO/'
SAVE_DIR = '{top_dir}/{sub}/'
SAVE_FILE = '{dir}TGM-LOSO-ANI_{sub}_win{win_len}_ov{ov}_pr{perm}_' \
            'alg{alg}_adj-{adj}_avgTime{avgTm}_avgTest{avgTst}_ni{inst}_' \
            'rsPerm{rsP}_{mode}'

FOLDS = range(32) #, 'coef']
EXPERIMENTS = ['krns2']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
WIN_LENS = [100]
OVERLAPS = [12]
IS_PERMS = [False]  # True
ALGS = ['lr-l2']  # GNB
ADJS = ['zscore']
DO_TIME_AVGS = [False]
DO_TEST_AVGS = [True]#, True]  # True
NUM_INSTANCESS = [10]
RANDOM_STATES = [1]

JOB_NAME = '{exp}-{sub}-{fold}-{id}'
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
    qsub_call = 'qsub -q pool2 -N {job_name} -l walltime=168:00:00,mem=8GB -v ' \
                'experiment={exp},subject={sub},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},fold={fold},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_ani_fold.sh'

    param_grid = itertools.product(FOLDS,
                                   EXPERIMENTS,
                                   OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_TIME_AVGS,
                                   DO_TEST_AVGS,
                                   NUM_INSTANCESS,
                                   RANDOM_STATES,
                                   WIN_LENS,
                                   SUBJECTS)
    job_id = 0
    for grid in param_grid:
        fold = grid[0]
        exp = grid[1]
        overlap = grid[2]
        isPerm = grid[3]
        alg = grid[4]
        adj = grid[5]
        tm_avg = grid[6]
        tst_avg = grid[7]
        ni = grid[8]
        rs = grid[9]
        win_len = grid[10]
        sub = grid[11]

        job_str = JOB_NAME.format(exp=exp,
                                  sub=sub,
                                  fold=fold,
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=exp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    exp=exp,
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
        top_dir = TOP_DIR.format(exp=exp)
        save_dir = SAVE_DIR.format(top_dir=top_dir, sub=sub)
        old_job = SAVE_FILE.format(dir=save_dir,
                                 sub=sub,
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
        if not os.path.isfile(old_job + '.npz'):
            call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 200:
            time.sleep(30)
