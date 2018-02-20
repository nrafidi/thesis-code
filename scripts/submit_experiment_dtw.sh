#!/usr/bin/env bash
source /home/python27/envs/p27default/bin/activate


cd /home/nrafidi/thesis-code/python
python dtw_comp.py --experiment $experiment  --subject $subject --sen_type $sen_type \
--dist $dist --radius $radius --tmax $tmax --sensors $sensors --num_instances $num_instances
