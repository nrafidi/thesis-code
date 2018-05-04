#!/usr/bin/env bash
source /home/python27/envs/p27default/bin/activate


cd /home/nrafidi/thesis-code/python
echo $force
python run_TGM_LOSO_det.py --subject $subject --sen_type $sen_type --analysis $analysis --win_len $win_len --overlap $overlap \
--isPerm $isPerm --adj $adj --alg $alg --doTimeAvg $doTimeAvg --doTestAvg $doTestAvg --mode $mode \
--num_instances $num_instances --perm_random_state $perm_random_state --force $force \
