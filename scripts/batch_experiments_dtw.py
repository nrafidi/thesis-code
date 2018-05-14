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
SUBJECTS = ['B']
RADIUS = [1]
TMINS = [0.0, 0.1, 0.2, 0.3]
TLENS = [0.05, 0.1, 0.5]
SEN0S = range(16)
SENSORS = [-1] # + range(306)
NINSTS = [10, 2]
METRICS = ['dtw', 'total']
VOICES = ['active']
DISTS = ['cosine']

JOB_NAME = '{sen0}-{radius}-{ninst}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

# JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'
JOB_Q_CHECK = 'expr $(qselect -q pool2 -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'

if __name__ == '__main__':

    qsub_call = 'qsub  -q pool2 -N {job_name} -l walltime=72:00:00,mem=4GB -v ' \
                'experiment={exp},subject={sub},radius={radius},sen0={sen0},voice={voice},metric={metric},' \
                'dist={dist},tmin={tmin},time_len={time_len},sensor={sensor},num_instances={ninst},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_dtw.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   SUBJECTS,
                                   VOICES,
                                   METRICS,
                                   RADIUS,
                                   TMINS,
                                   TLENS,
                                   SEN0S,
                                   NINSTS,
                                   DISTS,
                                   SENSORS)
    job_id = 0
    for grid in param_grid:
        exp = grid[0]
        sub = grid[1]
        voice = grid[2]
        metric = grid[3]
        radius = grid[4]
        tmin = grid[5]
        time_len = grid[6]
        sen0 = grid[7]
        ninst = grid[8]
        dist = grid[9]
        sensor = grid[10]

        job_str = JOB_NAME.format(sen0=sen0,
                                  radius=radius,
                                  ninst=ninst,
                                  id=job_id)

        dir_str = JOB_DIR.format(exp=exp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
        out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    exp=exp,
                                    sub=sub,
                                    voice=voice,
                                    metric=metric,
                                    sen0=sen0,
                                    radius=radius,
                                    dist=dist,
                                    tmin=tmin,
                                    time_len=time_len,
                                    sensor=sensor,
                                    ninst=ninst,
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        if job_id > 95:
            call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
