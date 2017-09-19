#!/bin/bash

for word in verb
do
    for subject in B C D E F G H
    do
	for win_len in  12 25 50 100 150 200 250 300 350
	do
	    for zscore in 0 #1
	    do
		qsub  -q default  -N pooled-$subject-$win_len-$zscore-$word-pdtw -v analysis=pooled_krns2_lr_pdtw,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=pooled,doZscore=$zscore,ddof=1,doAvg=0 submit_experiment_lr_pdtw.sh
		qsub  -q default  -N pooled-$subject-$win_len-$zscore-$word -v analysis=pooled_krns2_lr,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=pooled,doZscore=$zscore,ddof=1,doAvg=0,do2Samp=0 submit_experiment_lr.sh
		for actORpass in active passive
		do
		    qsub  -q default  -N $actORpass-$subject-$win_len-$zscore-$word-pdtw -v analysis=actORpass_krns2_lr_pdtw,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=$actORpass,doZscore=$zscore,ddof=1,doAvg=0 submit_experiment_lr_pdtw.sh    
		    qsub  -q default  -N $actORpass-$subject-$win_len-$zscore-$word -v analysis=actORpass_krns2_lr,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=$actORpass,doZscore=$zscore,ddof=1,doAvg=0,do2Samp=0 submit_experiment_lr.sh    
		done
	    done
	done
    done
done

#-l nodes=1:ppn=2
#	      qsub  -q default -N pooled_$subject-$win_len-$zscore-$word-$numFeats -v analysis=pooled_krns2,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=pooled,doZscore=$zscore,ddof=1,numFeats=$numFeats submit_experiment.sh