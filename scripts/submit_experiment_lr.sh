source /home/python27/envs/p27default/bin/activate
#export PYTHONPATH=/home/dhowarth/hippocampus/
cd /home/nrafidi/TGM_scripts
python run_experiment_2F_lr.py --analysis $analysis --experiment $experiment  --subject $sub --word $word --win_len $win_len --actORpass $actORpass --doZscore $doZscore --ddof $ddof  --doAvg $doAvg --do2Samp $do2Samp
