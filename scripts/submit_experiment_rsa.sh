#!/usr/bin/env bash
source /home/python27/envs/p27default/bin/activate

cd /home/nrafidi/thesis-code/python
python run_slide_noise_RSA.py --experiment $experiment  --subject $subject --word $word --win_len $win_len --overlap $overlap \
--dist $dist --doTimeAvg $doTimeAvg --draw $draw --force $force \
