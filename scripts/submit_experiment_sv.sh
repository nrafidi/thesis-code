#!/bin/bash
source /home/python27/envs/p27default/bin/activate

cd /home/nrafidi/thesis-code/python

python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
--isPerm $isPerm --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
--reps_to_use $reps_to_use --perm_random_state $perm_random_state --force $force --doTestAvg $doTestAvg