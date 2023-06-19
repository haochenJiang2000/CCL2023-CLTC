#!/bin/bash
#SBATCH -J hcjiang
#SBATCH -o /public/home/zhli13/hcjiang/MuCGEC-main/hcjiang_finetune.out
#SBATCH --gres=gpu:1

source /public/home/zhli13/anaconda3/bin/activate
conda activate seq2edit

# Step1. Data Preprocessing

## Download Structbert
if [ ! -f ./plm/chinese-struct-bert-large/pytorch_model.bin ]; then
    wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model
    mv ch_model ./plm/chinese-struct-bert-large/pytorch_model.bin
fi


## Tokenize
# DEV Set
SRC_FILE=../../data/learner/CCL2023/track1/dev/yaclc-minimal_dev_100/src.txt  # 每行一个病句
TGT_FILE=../../data/learner/CCL2023/track1/dev/yaclc-minimal_dev_100/tgt.txt  # 每行一个正确句子，和病句一一对应

if [ ! -f $SRC_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py --data_file $SRC_FILE --char_file $SRC_FILE".char"  # 分字
fi
if [ ! -f $TGT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py --data_file $TGT_FILE --char_file $TGT_FILE".char"  # 分字
fi

## Generate label file
LABEL_FILE=../../data/learner/CCL2023/track1/dev/yaclc-minimal_dev_100/dev_100.label  # 训练数据
if [ ! -f $LABEL_FILE ]; then
    python ./utils/preprocess_data.py \
    -s $SRC_FILE".char" \
    -t $TGT_FILE".char" \
    -o $LABEL_FILE \
    --worker_num 32 \

    shuf $LABEL_FILE > $LABEL_FILE".shuf"
fi

#Train Set
SRC_FILE=../../data/learner/CCL2023/track1/train/yaclc_dev/yaclc-minimal_dev_1739/src.txt # 每行一个病句
TGT_FILE=../../data/learner/CCL2023/track1/train/yaclc_dev/yaclc-minimal_dev_1739/tgt.txt  # 每行一个正确句子，和病句一一对应
if [ ! -f $SRC_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py --data_file $SRC_FILE --char_file $SRC_FILE".char"  # 分字
fi
if [ ! -f $TGT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py --data_file $TGT_FILE --char_file $TGT_FILE".char"  # 分字
fi

## Generate label file
LABEL_FILE=../../data/learner/CCL2023/track1/train/yaclc_dev/yaclc-minimal_dev_1739/dev_1739.label  # 训练数据
if [ ! -f $LABEL_FILE ]; then
    python ./utils/preprocess_data.py \
     -s $SRC_FILE".char" \
     -t $TGT_FILE".char" \
     -o $LABEL_FILE \
     --worker_num 32 \
     --save_vocab \
     --vocab_path "data/yaclc" \
     --min_count 1 \

    shuf $LABEL_FILE > $LABEL_FILE".shuf"
fi

# Step2. Training
CUDA_DEVICE=0
SEED=2023

DEV_SET=../../data/learner/CCL2023/track1/dev/yaclc-minimal_dev_100/dev_100.label.shuf
MODEL_DIR=./exps/seq2edit_lang8_CCL_open_finetune
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

PRETRAIN_WEIGHTS_DIR=./plm/chinese-struct-bert-large

mkdir ${MODEL_DIR}/src_bak
cp ./pipeline.sh $MODEL_DIR/src_bak
cp -r ./gector $MODEL_DIR/src_bak
cp ./train.py $MODEL_DIR/src_bak
cp ./predict.py $MODEL_DIR/src_bak

#VOCAB_PATH=./data/output_vocabulary_chinese_char_hsk+lang8_5
VOCAB_PATH=./data/output_vocabulary_chinese_learner_mix_pseudo

# Freeze encoder (Cold Step)
#COLD_LR=1e-3
#COLD_BATCH_SIZE=128
#COLD_MODEL_NAME=Best_Model_Stage_1
#COLD_EPOCH=2
#
#CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 0\
#                --train_set $LABEL_FILE".shuf"\
#                --dev_set $DEV_SET\
#                --model_dir $MODEL_DIR\
#                --model_name $COLD_MODEL_NAME\
#                --vocab_path $VOCAB_PATH\
#                --batch_size $COLD_BATCH_SIZE\
#                --n_epoch $COLD_EPOCH\
#                --lr $COLD_LR\
#                --weights_name $PRETRAIN_WEIGHTS_DIR\
#                --seed $SEED

## Unfreeze encoder
LR=1e-5
BATCH_SIZE=32
ACCUMULATION_SIZE=4
MODEL_NAME=Best_Model_Stage_2
EPOCH=20
PATIENCE=3

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 1\
                --train_set $LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $BATCH_SIZE\
                --n_epoch $EPOCH\
                --lr $LR\
                --accumulation_size $ACCUMULATION_SIZE\
                --patience $PATIENCE\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --pretrain_folder $MODEL_DIR\
                --pretrain "lert_2023"\
                --predictor_dropout 0.4\
                --seed $SEED\


#                --pretrain_folder $MODEL_DIR\
#                --pretrain "Pseudo_Best_Model_Stage_2"\

## Step3. Inference
#MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_2.th"
#RESULT_DIR=$MODEL_DIR"/results"
#
#INPUT_FILE=../../data/learner/MuCGEC_exp_data/MuCGEC_exp_data/test/MuCGEC.input # 输入文件
#if [ ! -f $INPUT_FILE".char" ]; then
#    python ../../tools/segment/segment_bert.py --data_file $INPUT_FILE --char_file $INPUT_FILE".char"  # 分字
#fi
#if [ ! -d $RESULT_DIR ]; then
#  mkdir -p $RESULT_DIR
#fi
#OUTPUT_FILE=$RESULT_DIR"/MuCGEC_test.output"
#
#echo "Generating..."
#SECONDS=0
#CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
#                  --weights_name $PRETRAIN_WEIGHTS_DIR\
#                  --vocab_path $VOCAB_PATH\
#                  --input_file $INPUT_FILE".char"\
#                  --output_file $OUTPUT_FILE --log
#
#echo "Generating Finish!"
#duration=$SECONDS
#echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "finish"
