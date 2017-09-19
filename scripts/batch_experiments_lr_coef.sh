#!/bin/bash
#actORpass=pooled

for subject in B C D E F G H
do
    for word in firstNoun verb secondNoun
    do
	for do2Samp in 0 #0 1
	do
	    for win_len in  12 25 50 100 150 200 250 300 350
	    do
		for doAvg in 0 #1
		do
		    for zscore in 0 #1
		    do
			qsub  -q default -N coef-pooled-$subject-$win_len-$zscore-$word -v analysis=pooled_krns2_lr_coef,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=meow,doZscore=$zscore,ddof=1,doAvg=$doAvg,do2Samp=$do2Samp submit_experiment_lr_coef.sh
			for actORpass in active passive
			do
			    qsub  -q default  -N coef-$actORpass-$subject-$win_len-$zscore-$word -v analysis=actORpass_krns2_lr_coef,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=$actORpass,doZscore=$zscore,ddof=1,doAvg=$doAvg,do2Samp=$do2Samp submit_experiment_lr_coef.sh
			done
		    done
		done
	    done
	done
    done
done

#-l nodes=1:ppn=2
#	      qsub  -q default -N pooled_$subject-$win_len-$zscore-$word-$numFeats -v analysis=pooled_krns2,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=pooled,doZscore=$zscore,ddof=1,numFeats=$numFeats submit_experiment.sh