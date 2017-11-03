import itertools
import os.path
from subprocess import call
import time

EXPERIMENTS = ['krns2']  # ,  'PassAct2', 'PassAct3']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
SEN_TYPES = ['active', 'passive'] #, 'active']
WORDS = ['firstNoun', 'verb', 'secondNoun']
IS_PDTWS = [False]  # True
IS_PERMS = [False]  # True
NUM_FOLDSS = [16, 32, 160]
ALGS = ['ridge']  # GNB
ADJS = [None, 'mean_center', 'zscore']
NUM_INSTANCESS = [1, 2, 10]  # 5 10
REPS_TO_USES = [10]  # 10
RANDOM_STATES = [1]  # range(1, 10)

JOB_NAME = 'sv-{exp}-{sub}-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_sv_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
                'experiment={exp},subject={sub},sen_type={sen},word={word},' \
                'isPDTW={pdtw},isPerm={perm},num_folds={nf},alg={alg},adj={adj},' \
                'num_instances={inst},reps_to_use={rep},perm_random_state={rs},force=True ' \
                '-e {errfile} -o {outfile} submit_experiment_sv.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   SUBJECTS,
                                   SEN_TYPES,
                                   WORDS,
                                   IS_PDTWS,
                                   IS_PERMS,
                                   NUM_FOLDSS,
                                   ALGS,
                                   ADJS,
                                   NUM_INSTANCESS,
                                   REPS_TO_USES,
                                   RANDOM_STATES)
    job_id = 0
    for grid in param_grid:
        job_str = JOB_NAME.format(exp=grid[0],
                                  sub=grid[1],
                                  sen=grid[2],
                                  word=grid[3],
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=grid[0])
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(exp=grid[0],
                                    sub=grid[1],
                                    sen=grid[2],
                                    word=grid[3],
                                    job_name=job_str,
                                    pdtw=grid[4],
                                    perm=grid[5],
                                    nf=grid[6],
                                    alg=grid[7],
                                    adj=grid[8],
                                    inst=grid[9],
                                    rep=grid[10],
                                    rs=grid[11],
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        call(call_str, shell=True)
        job_id += 1
        if job_id % 100 == 0:
            time.sleep(600)
