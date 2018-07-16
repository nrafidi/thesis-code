import itertools
import os.path
from subprocess import call, check_output
import time

def bool_to_str(bool_var):
    if bool_var:
        return 'T'
    else:
        return 'F'


FOLDS = range(32)
SEN_TYPES = ['pooled']
WORDS = ['verb', 'voice', 'agent', 'patient', 'propid']
WIN_LENS = [50]
OVERLAPS = [5]
IS_PERMS = [False]
ALGS = ['lr-l2']
ADJS = ['zscore']
DO_TME_AVGS = [True]
DO_TST_AVGS = [True]
NUM_INSTANCESS = [2]
RANDOM_STATES = [1]
REGIONS = ['superiorfrontal', 'caudalmiddlefrontal', 'rostralmiddlefrontal', 'parsopercularis', 'parsorbitalis',
              'parstriangularis', 'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole', 'paracentral',
              'precentral', 'insula', 'postcentral', 'inferiorparietal', 'supramarginal', 'superiorparietal',
              'precuneus', 'cuneus', 'lateraloccipital', 'lingual', 'pericalcarine', 'isthmuscingulate',
              'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate', 'entorhinal',
              'parahippocampal', 'temporalpole', 'fusiform', 'superiortemporal', 'inferiortemporal', 'middletemporal',
              'transversetemporal', 'bankssts', 'corpuscallosum']

HEMIS = ['lh', 'rh']

JOB_NAME = 'PA3-source-{sen}-{word}-{id}'
JOB_DIR = '/share/volume0/nrafidi/PassAct3_jobFiles/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

JOB_Q_CHECK = 'expr $(qselect -q default -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'
# JOB_Q_CHECK = 'expr $(qselect -q pool2 -u nrafidi | xargs qstat -u nrafidi | wc -l) - 5'

if __name__ == '__main__':

    qsub_call = 'qsub  -q default -N {job_name} -l walltime=192:00:00,mem=32GB -v ' \
                'sen_type={sen},word={word},win_len={win_len},overlap={overlap},region={reg},' \
                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg},fold={fold},hemi={hemi},' \
                'doTestAvg={tst_avg},num_instances={inst},perm_random_state={rs},force=False, ' \
                '-e {errfile} -o {outfile} submit_experiment_eos_multisub_fold_source.sh'

    param_grid = itertools.product(OVERLAPS,
                                   IS_PERMS,
                                   ALGS,
                                   ADJS,
                                   DO_TME_AVGS,
                                   DO_TST_AVGS,
                                   NUM_INSTANCESS,
                                   RANDOM_STATES,
                                   WIN_LENS,
                                   HEMIS,
                                   REGIONS,
                                   SEN_TYPES,
                                   WORDS,
                                   FOLDS)
    job_id = 0
    for grid in param_grid:
        overlap = grid[0]
        isPerm = grid[1]
        alg = grid[2]
        adj = grid[3]
        tm_avg = grid[4]
        tst_avg = grid[5]
        ni = grid[6]
        rs = grid[7]
        win_len = grid[8]
        hemi = grid[9]
        reg = grid[10]
        sen = grid[11]
        word = grid[12]
        fold = grid[13]

        if fold > 15 and sen != 'pooled':
            continue

        if word in ['propid', 'voice', 'senlen', 'noun1'] and sen != 'pooled':
            continue

        if word in ['agent', 'patient', 'propid']:
            if sen != 'pooled' and fold > 7:
                continue
            elif sen == 'pooled' and fold > 15:
                continue

        job_str = JOB_NAME.format(sen=sen,
                                  word=word,
                                  id=job_id)

        err_str = ERR_FILE.format(dir=JOB_DIR, job_name=job_str)
        out_str = OUT_FILE.format(dir=JOB_DIR, job_name=job_str)

        call_str = qsub_call.format(job_name=job_str,
                                    sen=sen,
                                    word=word,
                                    win_len=win_len,
                                    overlap=overlap,
                                    perm=isPerm,
                                    adj=adj,
                                    alg=alg,
                                    tm_avg=tm_avg,
                                    tst_avg=tst_avg,
                                    fold=fold,
                                    inst=ni,
                                    rs=rs,
                                    reg=reg,
                                    hemi=hemi,
                                    errfile=err_str,
                                    outfile=out_str)


        call(call_str, shell=True)
        job_id += 1

        while int(check_output(JOB_Q_CHECK, shell=True)) >= 100:
            time.sleep(30)
