#!/bin/bash

#parser.add_argument('--experiment')
#parser.add_argument('--subject')
#parser.add_argument('--sen_type')
#parser.add_argument('--word')
#parser.add_argument('--win_len', type=int)
#parser.add_argument('--overlap', type=int)
#parser.add_argument('--mode')
#parser.add_argument('--isPDTW', type=bool, default=False)
#parser.add_argument('--isPerm', type=bool, default=False)
#parser.add_argument('--num_folds', type=int, default=2)
#parser.add_argument('--alg', default='LR')
#parser.add_argument('--num_feats', type=int, default=500)
#parser.add_argument('--doZscore', type=bool, default=False)
#parser.add_argument('--doAvg', type=bool, default=False)
#parser.add_argument('--num_instances', type=int, default=2)
#parser.add_argument('--reps_to_use', type=int, default=10)
#parser.add_argument('--proc', default=load_data.DEFAULT_PROC)
#parser.add_argument('--random_state', type=int, default=1)


for experiment in krns2 #PassAct2, PassAct3
do
    for subject in B #C D E F G H
    do
        for sen_type in active #passive
        do
            for word in firstNoun #verb secondNoun
            do
                for win_len in  12 #12 25 50 100 150 200 250 300 350
                do
                    for overlap in 6 #12 25 50 $win_len
                    do
                        for mode in pred #coef
                        do
                            for isPDTW in False #True
                            do
                                for isPerm in False #True
                                do
                                    for num_folds in 2 #4 8
                                    do
                                        for alg in LR #GNB
                                        do
                                            for num_feats in 50 #100 150 200 500
                                            do
                                                for doZscore in False #True
                                                do
                                                    for doAvg in False #True
                                                    do
                                                        for num_instances in 2 #5 10
                                                        do
                                                            for reps_to_use in 15 #10
                                                            do
                                                                for random_state in 1 #2...
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
done
#-l nodes=1:ppn=2
#	      qsub  -q default -N pooled_$subject-$win_len-$zscore-$word-$numFeats -v analysis=pooled_krns2,experiment=krns2,sub=$subject,word=$word,win_len=$win_len,actORpass=pooled,doZscore=$zscore,ddof=1,numFeats=$numFeats submit_experiment.sh