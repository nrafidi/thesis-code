import itertools
import os.path
from subprocess import call, check_output
import time

# parser.add_argument('--experiment')
# parser.add_argument('--subject')
# parser.add_argument('--sen_type', choices=VALID_SEN_TYPE)
# parser.add_argument('--word', choices=['noun1', 'noun2', 'verb'])
# parser.add_argument('--win_len', type=int)
# parser.add_argument('--overlap', type=int)
# parser.add_argument('--isPerm', default='False', choices=['True', 'False'])
# parser.add_argument('--alg', default='ols', choices=VALID_ALGS)
# parser.add_argument('--adj', default='mean_center')
# parser.add_argument('--doTimeAvg', default='False', choices=['True', 'False'])
# parser.add_argument('--doTestAvg', default='False', choices=['True', 'False'])
# parser.add_argument('--num_instances', type=int, default=1)
# parser.add_argument('--reps_to_use', type=int, default=10)
# parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
# parser.add_argument('--perm_random_state', type=int, default=1)
# parser.add_argument('--force', default='False', choices=['True', 'False'])

EXPERIMENTS = ['krns2']  # ,  'PassAct2', 'PassAct3']
SUBJECTS = ['I', 'D', 'A', 'B', 'C', 'E', 'F', 'G', 'H']
SEN_TYPES = ['passive', 'active'] #, 'active']
WORDS = ['noun1', 'noun2', 'verb']
WIN_LENS = [-1, 12, 25, 50, 100]
OVERLAPS = [12, 25, 50, 100]
IS_PERMS = [False]  # True
ALGS = ['lr-l2', 'lr-l1']  # GNB
ADJS = [None, 'mean_center', 'zscore']
DO_AVGS = [True, False]  # True
NUM_INSTANCESS = [1, 2, 5, 10]
REPS_TO_USES = [10]  # 10
RANDOM_STATES = [1]

JOB_NAME = '{exp}-{sub}-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
                'experiment={exp},subject={sub},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg}' \
                'doTestAvg={tst_avg},num_instances={inst},reps_to_use={rep},perm_random_state={rs},force=False ' \
                '-e {errfile} -o {outfile} submit_experiment.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   SUBJECTS,
                                   SEN_TYPES,
                                   WORDS,
                                   WIN_LENS,
                                   OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_AVGS,
                                   DO_AVGS,
                                   NUM_INSTANCESS,
                                   REPS_TO_USES,
                                   RANDOM_STATES)
    job_id = 0
    for grid in param_grid:
        exp = grid[0]
        sub = grid[1]
        sen = grid[2]
        word = grid[3]
        win_len = grid[4]
        overlap = grid[5]
        isPerm = grid[6]
        alg = grid[7]
        adj = grid[8]
        tm_avg = grid[9]
        tst_avg = grid[10]
        ni = grid[11]
        reps = grid[12]
        rs = grid[13]

        job_str = JOB_NAME.format(exp=exp,
                                  sub=sub,
                                  sen=sen,
                                  word=word,
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=grid[0])
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    exp=exp,
                                    sub=sub,
                                    sen=sen,
                                    word=word,
                                    win_len=win_len,
                                    overlap=overlap,
                                    isPerm=isPerm,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    tst_avg=tst_avg,
                                    inst=ni,
                                    rep=reps,
                                    rs=rs,
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
