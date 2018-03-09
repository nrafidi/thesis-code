#!/usr/bin/env bash
source /home/python27/envs/p27default/bin/activate


cd /home/nrafidi/thesis-code/python
python dtw_sensor_select.py --experiment $experiment  --subject $subject --sen1 $sen1 \
--dist $dist --radius $radius --tmax $tmax --num_instances $num_instances
