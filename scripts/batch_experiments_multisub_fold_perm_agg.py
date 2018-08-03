from subprocess import call

JOB_NAME = 'agg-fold-perm'
JOB_DIR = '/share/volume0/nrafidi/'
ERR_FILE = '{dir}{job_name}.e'
OUT_FILE = '{dir}{job_name}.o'

if __name__ == '__main__':

    qsub_call = 'qsub -q default -N {job_name} -l walltime=168:00:00,mem=32GB ' \
                '-e {errfile} -o {outfile} submit_experiment_agg.sh'
    job_str = JOB_NAME
    dir_str = JOB_DIR

    err_str = ERR_FILE.format(dir=dir_str, job_name=job_str)
    out_str = OUT_FILE.format(dir=dir_str, job_name=job_str)

    call_str = qsub_call.format(job_name=job_str,
                                errfile=err_str,
                                outfile=out_str)

    call(call_str, shell=True)