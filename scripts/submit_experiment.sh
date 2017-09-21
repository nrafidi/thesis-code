#!/usr/bin/env bash
source /home/python27/envs/p27default/bin/activate

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


cd /home/nrafidi/thesis-code/python
echo $isPDTW
python run_TGM.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word --win_len $win_len --overlap $overlap --mode $mode \
--isPDTW false --isPerm $isPerm --num_folds $num_folds --alg $alg --num_feats $num_feats --doZscore $doZscore --doAvg $doAvg \
--num_instances $num_instances --reps_to_use $reps_to_use --random_state $random_state
