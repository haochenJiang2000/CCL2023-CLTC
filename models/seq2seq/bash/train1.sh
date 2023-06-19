#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --ntasks-per-node=1

source /public/home/swfeng/anaconda3/bin/activate
conda activate ccl23-cltc

SEED=2023
FAIRSEQ_CLI_PATH=../src/src_syngec/fairseq-0.10.2/fairseq_cli
MODEL_DIR_STAGE1=../model/pretrain/lang8_pseudo_v2/seed_$SEED
PROCESSED_DIR_STAGE1=../preprocess/lang8_pseudo_v2
FAIRSEQ_PATH=../src/src_syngec/fairseq-0.10.2/fairseq

mkdir -p $MODEL_DIR_STAGE1

mkdir -p $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE1/src

cp -r ../src/src_syngec/syngec_model $MODEL_DIR_STAGE1/src

cp ./train.sh $MODEL_DIR_STAGE1

# Transformer-base-setting stage 1

CUDA_VISIBLE_DEVICES=0 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE1/bin \
    --save-dir $MODEL_DIR_STAGE1 \
    --user-dir ../src/src_syngec/syngec_model \
    --bart-model-file-from-transformers ../pretrained_weights/bart-large-chinese \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_bart_large \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 16384 \
    --optimizer adam \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --lr 3e-05 \
    --warmup-updates 2000 \
    -s src \
    -t tgt \
    --lr-scheduler polynomial_decay \
    --clip-norm 1.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.999)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 5 \
    --seed $SEED >${MODEL_DIR_STAGE1}/nohup.log 2>&1 &

wait