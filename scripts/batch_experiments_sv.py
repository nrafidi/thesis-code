import itertools
import os.path
from subprocess import call, check_output
import time

EXPERIMENTS = ['krns2', 'PassAct2']  # ,  'PassAct2', 'PassAct3']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
SEN_TYPES = ['active', 'passive']
WORDS = ['all', 'noun1', 'verb', 'noun2']
IS_PERMS = [False]  # True
NUM_FOLDSS = [16] #, 32, 160]
ALGS = ['ols', 'ridge']  # GNB
ADJS = [None, 'mean_center', 'zscore']
NUM_INSTANCESS = [2, 10] #, 2, 10]  # 5 10
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
                'isPerm={perm},num_folds={nf},alg={alg},adj={adj},' \
                'num_instances={inst},reps_to_use={rep},perm_random_state={rs},force=False ' \
                '-e {errfile} -o {outfile} submit_experiment_sv.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   SUBJECTS,
                                   SEN_TYPES,
                                   WORDS,
                                   IS_PERMS,
                                   NUM_FOLDSS,
                                   ALGS,
                                   ADJS,
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
        nf = grid[5]
        alg = grid[6]
        adj = grid[7]
        inst = grid[8]
        rep = grid[9]
        rs = grid[10]


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
                                    nf=nf,
                                    alg=alg,
                                    adj=adj,
                                    inst=inst,
                                    rep=rep,
                                    rs=rs,
                                    job_name=job_str,
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
