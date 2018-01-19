import itertools
import os.path
from subprocess import call, check_output
import time

EXPERIMENTS = ['krns2', 'PassAct2']  # ,  'PassAct2', 'PassAct3']
SUBJECTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
SEN_TYPES = ['active', 'passive']
WORDS = ['all', 'noun1', 'verb', 'noun2']
IS_PERMS = [False]  # True
# NUM_FOLDSS = [16, 32, 80, 160]
ALGS = ['ols', 'ridge']  # GNB
ADJS = [None, 'mean_center', 'zscore']
TST_AVGS = [True, False]
NUM_INSTANCESS = [1, 2, 5, 10]
REPS_TO_USES = [10]  # 10
RANDOM_STATES = [1]

JOB_NAME = 'oh-{exp}-{sub}-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_oh_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qstat -u nrafidi | wc -l) - 5'

if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=32GB -v ' \
                'experiment={exp},subject={sub},sen_type={sen},word={word},' \
                'isPerm={perm},alg={alg},adjX={adjX},adjY={adjY},doTestAvg={tst_avg},' \
                'num_instances={inst},reps_to_use={rep},perm_random_state={rs},force=False ' \
                '-e {errfile} -o {outfile} submit_experiment_sv.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   SUBJECTS,
                                   SEN_TYPES,
                                   WORDS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   ADJS,
                                   TST_AVGS,
                                   NUM_INSTANCESS,
                                   REPS_TO_USES,
                                   RANDOM_STATES)
    job_id = 0
    for grid in param_grid:
        exp = grid[0]
        sub = grid[1]
        sen = grid[2]
        word = grid[3]
        perm = grid[4]
        alg = grid[5]
        adjX = grid[6]
        adjY = grid[7]
        tst_avg = grid[8]
        inst = grid[9]
        rep = grid[10]
        rs = grid[11]


        job_str = JOB_NAME.format(exp=exp,
                                  sub=sub,
                                  sen=sen,
                                  word=word,
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=exp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(exp=exp,
                                    sub=sub,
                                    sen=sen,
                                    word=word,
                                    perm=perm,
                                    alg=alg,
                                    adjX=adjX,
                                    adjY=adjY,
                                    tst_avg=tst_avg,
                                    inst=inst,
                                    rep=rep,
                                    rs=rs,
                                    job_name=job_str,
                                    errfile=err_str,
                                    outfile=out_str)
        #print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 40:
            time.sleep(30)
