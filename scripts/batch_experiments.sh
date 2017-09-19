#!/bin/bash

zscore=0

for subject in C #B C D E F G H
do
    for word in verb #firstNoun verb secondNoun
    do
	for actORpass in active passive
	do
	    for win_len in  12 #150 200 #12 25 50 100 #250 300 350
	    do
		for doAvg in 1
		do
		    if [ $doAvg == 1 ]
		    then
			for numFeats in 0 10 20 50 100 200
			do
			    qsub  -q default -N $actORpass-$subject-$win_len-$zscore-$word-$numFeats -v analysis=actORpass_krns2,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=$actORpass,doZscore=0,ddof=1,numFeats=$numFeats,doAvg=$doAvg submit_experiment.sh
			done
		    else
			for numFeats in 0 100 200 300 400 500 1000 1500 2000 2500 3000
			do
			    qsub  -q default  -N $actORpass-$subject-$win_len-$zscore-$word-$numFeats -v analysis=actORpass_krns2,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=$actORpass,doZscore=0,ddof=1,numFeats=$numFeats,doAvg=$doAvg submit_experiment.sh
			done
		    fi
		done
	    done
	done
    done
done

#-l nodes=1:ppn=2
#	      qsub  -q default -N pooled_$subject-$win_len-$zscore-$word-$numFeats -v analysis=pooled_krns2,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=pooled,doZscore=$zscore,ddof=1,numFeats=$numFeats submit_experiment.sh