#SBATCH -J train
#SBATCH -p batch
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1

source /public/home/swfeng/anaconda3/bin/activate
conda activate ccl23-cltc

for SEED in 6 21 22 23 24  
do
    FAIRSEQ_CLI_PATH=../src/src_syngec/fairseq-0.10.2/fairseq_cli
    MODEL_DIR_STAGE1=../model/finetune-ii/single-source-wo-lang8/seed_3023
    MODEL_DIR_STAGE2=../model/finetune-iii/yaclc-minimal_dev_1739/seed_$SEED
    PROCESSED_DIR_STAGE2=../preprocess/yaclc-minimal_dev_1739
    FAIRSEQ_PATH=../src/src_syngec/fairseq-0.10.2/fairseq

    mkdir -p $MODEL_DIR_STAGE2 && mkdir -p $MODEL_DIR_STAGE2/src

    cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE2/src

    cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE2/src

    cp -r ../src/src_syngec/syngec_model $MODEL_DIR_STAGE2/src

    cp ./finetune.sh $MODEL_DIR_STAGE2

    # Transformer-base-setting stage 2

    CUDA_VISIBLE_DEVICES=0 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE2/bin \
        --save-dir $MODEL_DIR_STAGE2 \
        --user-dir ../src/src_syngec/syngec_model \
        --finetune-from-model $MODEL_DIR_STAGE1/checkpoint_best.pt \
        --task syntax-enhanced-translation \
        --arch syntax_enhanced_bart_large \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 16384 \
        --optimizer adam \
        --max-source-positions 512 \
        --max-target-positions 512 \
        --lr 3e-6 \
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
        --no-epoch-checkpoints \
        --patience 5 \
        --seed $SEED >${MODEL_DIR_STAGE2}/nohup.log 2>&1 &

    wait

    CUDA_DEVICE=0
    BEAM=12
    N_BEST=1
    FAIRSEQ_DIR=../src/src_syngec/fairseq-0.10.2/fairseq_cli

    TEST_DIR=../data/yaclc-minimal_test
    MODEL_DIR=../model/finetune-iii/yaclc-minimal_dev_1739/seed_$SEED
    ID_FILE=$TEST_DIR/src.id
    PROCESSED_DIR=../preprocess/yaclc-minimal_dev_1739

    OUTPUT_DIR=$MODEL_DIR/results

    mkdir -p $OUTPUT_DIR
    cp $ID_FILE $OUTPUT_DIR/yaclc-minimal_test.id
    cp $TEST_DIR/src.txt.char $OUTPUT_DIR/yaclc-minimal_test.src.char

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
        --output_file $OUTPUT_DIR/yaclc-minimal_test.out.nbest \
        < $OUTPUT_DIR/yaclc-minimal_test.src.char

    echo "Generating Finish!"
    duration=$SECONDS
    echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

    cat $OUTPUT_DIR/yaclc-minimal_test.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/yaclc-minimal_test.out
    sed -i '$d' $OUTPUT_DIR/yaclc-minimal_test.out
    python ../utils/post_process_chinese.py $OUTPUT_DIR/yaclc-minimal_test.src.char $OUTPUT_DIR/yaclc-minimal_test.out $OUTPUT_DIR/yaclc-minimal_test.id $OUTPUT_DIR/yaclc-minimal_test.out.post_processed 7296

    CUDA_DEVICE=0
    BEAM=12
    N_BEST=1
    FAIRSEQ_DIR=../src/src_syngec/fairseq-0.10.2/fairseq_cli

    TEST_DIR=../data/yaclc-fluency_test
    MODEL_DIR=../model/finetune-iii/yaclc-minimal_dev_1739/seed_$SEED
    ID_FILE=$TEST_DIR/src.id
    PROCESSED_DIR=../preprocess/yaclc-minimal_dev_1739

    OUTPUT_DIR=$MODEL_DIR/results

    mkdir -p $OUTPUT_DIR
    cp $ID_FILE $OUTPUT_DIR/yaclc-fluency_test.id
    cp $TEST_DIR/src.txt.char $OUTPUT_DIR/yaclc-fluency_test.src.char

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
        --output_file $OUTPUT_DIR/yaclc-fluency_test.out.nbest \
        < $OUTPUT_DIR/yaclc-fluency_test.src.char

    echo "Generating Finish!"
    duration=$SECONDS
    echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

    cat $OUTPUT_DIR/yaclc-fluency_test.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/yaclc-fluency_test.out
    sed -i '$d' $OUTPUT_DIR/yaclc-fluency_test.out
    python ../utils/post_process_chinese.py $OUTPUT_DIR/yaclc-fluency_test.src.char $OUTPUT_DIR/yaclc-fluency_test.out $OUTPUT_DIR/yaclc-fluency_test.id $OUTPUT_DIR/yaclc-fluency_test.out.post_processed 5515
done