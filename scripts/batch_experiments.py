import itertools
import os.path
from subprocess import call
import time

#parser.add_argument('--experiment')
#parser.add_argument('--subject')
#parser.add_argument('--SEN_TYPE')
#parser.add_argument('--word')
#parser.add_argument('--win_len', type=int)
#parser.add_argument('--overlap', type=int)
#parser.add_argument('--mode')
#parser.add_argument('--isPDTW', type=bool, default=False)
#parser.add_argument('--isPerm', type=bool, default=False)
#parser.add_argument('--num_folds', type=int, default=2)
#parser.add_argument('--alg', default='LR')
#parser.add_argument('--doZscore', type=bool, default=False)
#parser.add_argument('--doAvg', type=bool, default=False)
#parser.add_argument('--num_instances', type=int, default=2)
#parser.add_argument('--reps_to_use', type=int, default=10)
#parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
#parser.add_argument('--random_state', type=int, default=1)

EXPERIMENTS = ['krns2']  # ,  'PassAct2', 'PassAct3']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
SEN_TYPES = ['active', 'passive'] #, 'active']
WORDS = ['firstNoun', 'verb', 'secondNoun']
WIN_LENS = [-1] #-1, 3, 6, 12, 25] #, 2000]
OVERLAPS = [3] #12, 25, 50, 100, 150, 200, 250, 300, 350]
MODES = ['uni']  # pred
IS_PDTWS = [False]  # True
IS_PERMS = [True]  # True
NUM_FOLDSS = [32]
ALGS = ['GNB']  # GNB
DO_ZSCORES = [False]  # True
DO_AVGS = [False]  # True
NUM_INSTANCESS = [2]  # 5 10
REPS_TO_USES = [10]  # 10
RANDOM_STATES = range(1, 100)

JOB_NAME = '{exp}-{sub}-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
                'experiment={exp},subject={sub},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'mode={mode},isPDTW={pdtw},isPerm={perm},num_folds={nf},alg={alg},doZscore={z},' \
                'doAvg={avg},num_instances={inst},reps_to_use={rep},perm_random_state={rs},force=False,doFeatSelect=True ' \
                '-e {errfile} -o {outfile} submit_experiment.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   SUBJECTS,
                                   SEN_TYPES,
                                   WORDS,
                                   WIN_LENS,
                                   OVERLAPS,
                                   MODES,
                                   IS_PDTWS,
                                   IS_PERMS,
                                   NUM_FOLDSS,
                                   ALGS,
                                   DO_ZSCORES,
                                   DO_AVGS,
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
                                    win_len=grid[4],
                                    overlap=grid[5],
                                    mode=grid[6],
                                    pdtw=grid[7],
                                    perm=grid[8],
                                    nf=grid[9],
                                    alg=grid[10],
                                    z=grid[11],
                                    avg=grid[12],
                                    inst=grid[13],
                                    rep=grid[14],
                                    rs=grid[15],
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        call(call_str, shell=True)
        job_id += 1
        if job_id % 100 == 0:
            time.sleep(300)
