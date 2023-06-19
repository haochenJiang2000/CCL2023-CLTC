TESTDIR=../data/yaclc-fluency_dev/single


python ../utils/segment_sent.py $TESTDIR/src.ori.txt $TESTDIR/src.txt $TESTDIR/src.id
python ../utils/segment_bert.py <$TESTDIR/src.txt >$TESTDIR/src.txt.char