
DATANAME=yaclc-fluency_test
TESTPATH=../data/$DATANAME/$DATANAME.src
MODELDIR=../model/finetune-i/yaclc-minimal_dev_1739/seed_7
RESULTPATH=$MODELDIR/results/$DATANAME.out.post_processed
SUBMITPATH=$MODELDIR/results/track_test/$DATANAME.para

python ../utils/submit.py $TESTPATH $RESULTPATH $SUBMITPATH

DATANAME=yaclc-minimal_test
TESTPATH=../data/$DATANAME/$DATANAME.src
RESULTPATH=$MODELDIR/results/$DATANAME.out.post_processed
SUBMITPATH=$MODELDIR/results/track_test/$DATANAME.para

python ../utils/submit.py $TESTPATH $RESULTPATH $SUBMITPATH