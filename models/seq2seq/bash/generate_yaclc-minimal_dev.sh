#!/bin/bash
#SBATCH -J gen
#SBATCH --gres=gpu:1
# source /public/home/swfeng/anaconda3/bin/activate
# conda activate ccl23-cltc

CUDA_DEVICE=0
BEAM=12
N_BEST=1
SEED=2023
FAIRSEQ_DIR=../src/src_syngec/fairseq-0.10.2/fairseq_cli

TEST_DIR=../data/yaclc-minimal_dev/single
MODEL_DIR=../model/finetune-i/lang8_v2/seed_$SEED
ID_FILE=$TEST_DIR/src.id
PROCESSED_DIR=../preprocess/lang8_v2

OUTPUT_DIR=$MODEL_DIR/results

mkdir -p $OUTPUT_DIR
cp $ID_FILE $OUTPUT_DIR/yaclc-minimal_dev.id
cp $TEST_DIR/src.txt.char $OUTPUT_DIR/yaclc-minimal_dev.src.char

echo "Generating ..."
SECONDS=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR/bin \
    --user-dir ../src/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --path ${MODEL_DIR}/checkpoint_best.pt \
    --beam ${BEAM} \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --buffer-size 10000 \
    --batch-size 32 \
    --num-workers 12 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --output_file $OUTPUT_DIR/yaclc-minimal_dev.out.nbest \
    < $OUTPUT_DIR/yaclc-minimal_dev.src.char

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/yaclc-minimal_dev.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/yaclc-minimal_dev.out
sed -i '$d' $OUTPUT_DIR/yaclc-minimal_dev.out
python ../utils/post_process_chinese.py $OUTPUT_DIR/yaclc-minimal_dev.src.char $OUTPUT_DIR/yaclc-minimal_dev.out $OUTPUT_DIR/yaclc-minimal_dev.id $OUTPUT_DIR/yaclc-minimal_dev.out.post_processed 1839
