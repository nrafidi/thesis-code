#!/usr/bin/env bash
source /home/python27/envs/p27default/bin/activate


cd /home/nrafidi/thesis-code/python
python dtw_sensor_select.py --experiment $experiment  --subject $subject --sen0 $sen0 \
--dist $dist --radius $radius --tmin $tmin --tmax $tmax --sensor $sensor --num_instances $num_instances
