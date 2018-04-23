import itertools
import os.path
from subprocess import call, check_output
import time



JOB_NAME = 'alg-time'
JOB_DIR = '/share/volume0/nrafidi/krns2_jobFiles/'
ERR_FILE = JOB_DIR + JOB_NAME + '.e'
OUT_FILE = JOB_DIR + JOB_NAME + '.o'

if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N ' + JOB_NAME + ' -l walltime=200:00:00,mem=2GB ' \
                '-e {errfile} -o {outfile} submit_experiment_alg.sh'

    call(qsub_call, shell=True)
