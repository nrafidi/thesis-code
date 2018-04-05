import itertools
import os.path
from subprocess import call, check_output
import time


EXPERIMENTS = ['PassAct3']
SUBJECTS = ['A', 'B', 'C', 'E', 'F', 'G', 'J', 'K', 'L', 'N', 'O', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z']
WORDS = ['det', 'noun2', 'eos', 'last-full', 'eos-full']
WIN_LENS = [50]
OVERLAPS = [3]
DRAWS = range(126)
DISTS = ['cosine', 'euclidean']
DO_TME_AVGS = [True]

JOB_NAME = '{dist}-{sub}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/{exp}_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q pool2 -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'


if __name__ == '__main__':

    qsub_call = 'qsub  -q pool2 -N {job_name} -v ' \
                'experiment={exp},subject={sub},word={word},win_len={win_len},overlap={overlap},' \
                'dist={dist},doTimeAvg={tm_avg},draw={draw},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_rsa.sh'

    param_grid = itertools.product(EXPERIMENTS,
                                   OVERLAPS,
                                   DISTS,
                                   DO_TME_AVGS,
                                   WORDS,
                                   WIN_LENS,
                                   SUBJECTS,
                                   DRAWS)
    job_id = 0
    for grid in param_grid:
        exp = grid[0]
        overlap = grid[1]
        dist = grid[2]
        tm_avg = grid[3]
        word = grid[4]
        win_len = grid[5]
        sub = grid[6]
        draw = grid[7]

        job_str = JOB_NAME.format(dist=dist,
                                  sub=sub,
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
                                    draw=draw,
                                    dist=dist,
                                    word=word,
                                    win_len=win_len,
                                    overlap=overlap,
                                    tm_avg=tm_avg,
                                    errfile=err_str,
                                    outfile=out_str)
        # print(call_str)
        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 400:
            time.sleep(30)
