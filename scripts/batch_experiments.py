import itertools
import os.path
from subprocess import call

#parser.add_argument('--experiment')
#parser.add_argument('--subject')
#parser.add_argument('--sen_type')
#parser.add_argument('--word')
#parser.add_argument('--win_len', type=int)
#parser.add_argument('--overlap', type=int)
#parser.add_argument('--mode')
#parser.add_argument('--isPDTW', type=bool, default=False)
#parser.add_argument('--isPerm', type=bool, default=False)
#parser.add_argument('--num_folds', type=int, default=2)
#parser.add_argument('--alg', default='LR')
#parser.add_argument('--num_feats', type=int, default=500)
#parser.add_argument('--doZscore', type=bool, default=False)
#parser.add_argument('--doAvg', type=bool, default=False)
#parser.add_argument('--num_instances', type=int, default=2)
#parser.add_argument('--reps_to_use', type=int, default=10)
#parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
#parser.add_argument('--random_state', type=int, default=1)

if __name__ == '__main__':
    experiments = ['krns2'] # PassAct2, PassAct3
    subjects = ['B'] # C D E F G H
    sen_type = ['active'] # passive']
    word = ['firstNoun']  #  verb secondNoun
    win_len = [12]  # 12 25 50 100 150 200 250 300 350
    overlap = [6]  # 12 25 50 100 150 200 250 300 350
    mode = ['pred']  # coef
    isPDTW = [False]  # True
    isPerm = [False]  # True
    num_folds = [2]  # 4 8
    alg = ['LR']  # GNB
    num_feats = [50]  # 100 150 200 500
    doZscore = [False]  # True
    doAvg = [False]  # True
    num_instances = [2]  # 5 10
    reps_to_use = [15]  # 10
    random_state = [1]  # range(1, 10)

    job_name = '{exp}-{sub}-{sen}-{word}-{id}'

    job_dir = '/share/volume0/nrafidi/{exp}_jobFiles/'
    err_file = '{dir}{job_name}.e'

    out_file = '{dir}{exp}_jobFiles/{job_name}.o'

    qsub_call = 'qsub  -q default -N {job_name} -v ' \
                'experiment={exp},subject={sub},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
                'mode={mode},isPDTW={pdtw},isPerm={perm},num_folds={nf},alg={alg},num_feats={num_feats},doZscore={z},' \
                'doAvg={avg},num_instances={inst},reps_to_use={rep},random_state={rs} ' \
                '-e {errfile} -o {outfile} submit_experiment.sh'

    param_grid = itertools.product(experiments,
                                   subjects,
                                   sen_type,
                                   word,
                                   win_len,
                                   overlap,
                                   mode,
                                   isPDTW,
                                   isPerm,
                                   num_folds,
                                   alg,
                                   num_feats,
                                   doZscore,
                                   doAvg,
                                   num_instances,
                                   reps_to_use,
                                   random_state)
    job_id = 0
    for grid in param_grid:
        job_str = job_name.format(exp=grid[0],
                                  sub=grid[1],
                                  sen=grid[2],
                                  word=grid[3],
                                  id=job_id)

        dir_str = job_dir.format(exp=grid[0])
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        err_str = err_file.format(dir=dir_str, job_name=job_str)
        out_str = out_file.format(dir=dir_str, job_name=job_str)

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
                                    num_feats=grid[11],
                                    z=grid[12],
                                    avg=grid[13],
                                    inst=grid[14],
                                    rep=grid[15],
                                    rs=grid[16],
                                    errfile=err_str,
                                    outfile=out_str)
        call(call_str, shell=True)
        job_id += 1
