#!/usr/bin/bash
source /home/python27/envs/p27default/bin/activate

#parser.add_argument('--experiment')
#parser.add_argument('--subject')
#parser.add_argument('--sen_type')
#parser.add_argument('--word')
#parser.add_argument('--model', default='one_hot')
#    parser.add_argument('--inc_art1', default='False')
#    parser.add_argument('--inc_art2', default='False')
#    parser.add_argument('--only_art1', default='False')
#    parser.add_argument('--only_art2', default='False')
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


cd /home/nrafidi/thesis-code/python

echo $doTestAvg
echo 'Dan is a poop head'
echo 'Dan is a major poop head'
echo 'It might have been defaulting'
if $doTestAvg==True; then
    echo 'meow'
elif $doTestAvg=='True'; then
    echo 'woof'
elif $doTestAvg=="True"; then
    echo 'oink'
elif $doTestAvg; then
    echo 'moo'
elif "$doTestAvg"=="True"; then
    echo 'bark'
elif "$doTestAvg"==True; then
    echo 'ruff'
elif "$doTestAvg"==true; then
    echo 'Calvin is a poop head'
fi
echo $force
if $force==True
then
    echo 'meow'
elif $force=='True'
then
    echo 'woof'
fi

if $isPerm =='True' and $force == 'True' and $doTestAvg == 'True'
then
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --isPerm --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state --force --doTestAvg
elif $isPerm == 'True' and $force == 'True'
then
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --isPerm --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state --force
elif $isPerm == 'True' and $doTestAvg == 'True'
then
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --isPerm --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state --doTestAvg
elif $isPerm == 'True'
then
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --isPerm --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state
elif $force == 'True' and $doTestAvg == 'True'
then
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state --force --doTestAvg
elif $force == 'True'
then
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state --force
elif $doTestAvg == 'True'
then
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state --doTestAvg
else
    python run_OH_Reg_LOSO.py --experiment $experiment  --subject $subject --sen_type $sen_type --word $word \
    --alg $alg --adjX $adjX --adjY $adjY --num_instances $num_instances \
    --reps_to_use $reps_to_use --perm_random_state $perm_random_state
fi