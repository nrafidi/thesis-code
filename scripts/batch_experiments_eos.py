import itertools
import os.path
from subprocess import call, check_output
import time


MODES = ['acc']
EXPERIMENTS = ['krns2']#, 'PassAct3']
SUBJECTS = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] #['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N']
SEN_TYPES = ['pooled']
WORDS = ['verb', 'noun1', 'voice', 'agent', 'patient', 'propid']
WIN_LENS = [12, 25, 50, 100]
OVERLAPS = [12]
IS_PERMS = [False]
ALGS = ['lr-l2']
ADJS = ['zscore']
DO_TME_AVGS = [False]
DO_TST_AVGS = [True]
NUM_INSTANCESS = [1, 2, 5, 10]
RANDOM_STATES = [1]

JOB_NAME = '{exp}-{sub}-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=192:00:00,mem=8GB -v ' \
                'experiment={exp},subject={sub},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},mode={mode},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_eos.sh'

    param_grid = itertools.product(MODES,
                                   EXPERIMENTS,
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
        rs = grid[9]
        win_len = grid[10]
        sen = grid[11]
        word = grid[12]
        sub = grid[13]

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
                                    rs=rs,
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        if job_id in [716, 717, 718, 719, 721, 724, 728, 729, 730,
                      731, 732, 733, 734, 735]:
            call(call_str, shell=True)
        job_id += 1

        # while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
        #     time.sleep(30)
