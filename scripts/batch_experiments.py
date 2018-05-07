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

MODES = ['acc'] #, 'coef']
EXPERIMENTS = ['krns2'] #, 'PassAct3']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] #['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']
SEN_TYPES = ['active', 'passive'] #, 'active']
WORDS = ['noun1', 'verb', 'noun2']
WIN_LENS = [12, 25, 50, 100, 150]
OVERLAPS = [12]
IS_PERMS = [False]  # True
ALGS = ['lr-l2']  # GNB
ADJS = ['zscore']
DO_TIME_AVGS = [False]
DO_TEST_AVGS = [True]#, True]  # True
NUM_INSTANCESS = [1, 2, 5, 10]
REPS_TO_USES = [None]  # 10
RANDOM_STATES = [1]

JOB_NAME = '{exp}-{sub}-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
                'experiment={exp},subject={sub},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},mode={mode},' \
                'doTestAvg={tst_avg},num_instances={inst},reps_to_use={rep},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment.sh'

    param_grid = itertools.product(MODES,
                                   EXPERIMENTS,
                                   OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_TIME_AVGS,
                                   DO_TEST_AVGS,
                                   NUM_INSTANCESS,
                                   REPS_TO_USES,
                                   RANDOM_STATES,
                                   WIN_LENS,
                                   SEN_TYPES,
                                   WORDS,
                                   SUBJECTS)
    job_id = 0
    for grid in param_grid:
        mode = grid[0]
        exp = grid[1]
        overlap = grid[2]
        isPerm = grid[3]
        alg = grid[4]
        adj = grid[5]
        tm_avg = grid[6]
        tst_avg = grid[7]
        ni = grid[8]
        reps = grid[9]
        rs = grid[10]
        win_len = grid[11]
        sen = grid[12]
        word = grid[13]
        sub = grid[14]

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

        call_str = qsub_call.format(job_name=job_str,
                                    exp=exp,
                                    sub=sub,
                                    sen=sen,
                                    word=word,
                                    win_len=win_len,
                                    overlap=overlap,
                                    perm=isPerm,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    tst_avg=tst_avg,
                                    mode=mode,
                                    inst=ni,
                                    rep=reps,
                                    rs=rs,
                                    errfile=err_str,
                                    outfile=out_str)
        print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
