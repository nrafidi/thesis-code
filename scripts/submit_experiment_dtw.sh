#!/usr/bin/env bash
source /home/python27/envs/p27default/bin/activate


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

#qsub_call = 'qsub  -q default -N {job_name} -l walltime=72:00:00,mem=2GB -v ' \
#                'experiment={exp},subject={sub},sen_type={sen},word={word},win_len={win_len},overlap={overlap},' \
#                'isPerm={perm},adj={adj},alg={alg},doTimeAvg={tm_avg}' \
#                'doTestAvg={tst_avg},num_instances={inst},reps_to_use={rep},perm_random_state={rs},force=False ' \
#                '-e {errfile} -o {outfile} submit_experiment.sh'

cd /home/nrafidi/thesis-code/python
python dtw_comp.py --experiment $experiment  --subject $subject --sen_type $sen_type \
--dist $dist --radius $radius
