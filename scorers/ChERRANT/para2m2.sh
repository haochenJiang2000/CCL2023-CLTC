#DATA_FILE=../../data/MuCGEC2.0/MuCGEC2.0/paper
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.word" -g word


#DATA_FILE=../../data/MuCGEC2.0/MuCGEC2.0/bingju
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.word" -g word


#DATA_FILE=../../data/MuCGEC2.0/MuCGEC2.0/weixin
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/train/train_notag.para" -o $DATA_FILE"/train/train.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/valid/valid_notag.para" -o $DATA_FILE"/valid/valid.m2.word" -g word
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.char" -g char
#python parallel_to_m2.py -f $DATA_FILE"/test/test_notag.para" -o $DATA_FILE"/test/test.m2.word" -g word

#python parallel_to_m2.py -f total_notag.para -o total.m2.word -g word
#python parallel_to_m2.py -f total_notag.para -o total.m2.char -g char


#python parallel_to_m2.py -f "yaclc_minimal_dev.predict.ensemble.para" -o "yaclc_minimal_dev.predict.ensemble.m2.char" -g char
#python compare_m2_for_evaluation.py -hyp "yaclc_minimal_dev.predict.ensemble.m2.char" -ref "evaluate_data/yaclc-minimal_dev.m2"

#python parallel_to_m2.py -f "evaluate_data/thesis.para" -o "evaluate_data/thesis.m2.char" -g char
#python compare_m2_for_evaluation.py -hyp "evaluate_data/thesis.m2.char" -ref "evaluate_data/thesis.m2"

python parallel_to_m2.py -f "evaluate_data/test.para" -o "evaluate_data/test.m2.char" -g char