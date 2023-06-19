#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --ntasks-per-node=1

for SEED in 2023 3023 4023 5023 6023 7023
do
    source /public/home/swfeng/anaconda3/bin/activate
    conda activate ccl23-cltc
    FAIRSEQ_CLI_PATH=../src/src_syngec/fairseq-0.10.2/fairseq_cli
    MODEL_DIR_STAGE1=../model/finetune-i/lang8+hsk+mucgec_dev+nlpcc18_test+cged/seed_$SEED
    MODEL_DIR_STAGE2=../model/finetune-ii/hsk+mucgec_dev+nlpcc18_test+cged/seed_$SEED
    PROCESSED_DIR_STAGE2=../preprocess/hsk+mucgec_dev+nlpcc18_test+cged
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
        --lr 5e-6 \
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

    TEST_DIR=../data/yaclc-minimal_dev/single
    MODEL_DIR=../model/finetune-ii/hsk+mucgec_dev+nlpcc18_test+cged/seed_$SEED
    ID_FILE=$TEST_DIR/src.id
    PROCESSED_DIR=../preprocess/hsk+mucgec_dev+nlpcc18_test+cged

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

    source /public/home/swfeng/anaconda3/bin/activate
    conda activate cherrant
    MODEL_DIR=../model/finetune-ii/hsk+mucgec_dev+nlpcc18_test+cged/seed_$SEED
    OUTPUT_DIR=$MODEL_DIR/results
    HYP_PARA_FILE=$OUTPUT_DIR/yaclc-minimal_dev.out.para
    HYP_M2_FILE=$OUTPUT_DIR/yaclc-minimal_dev.out.m2
    DATA_DIR=../data/yaclc-minimal_dev
    REF_M2_FILE=$DATA_DIR/yaclc-minimal_dev.m2

    paste $DATA_DIR/single/src.txt $OUTPUT_DIR/yaclc-minimal_dev.out.post_processed | awk '{print NR"\t"$p}' > $HYP_PARA_FILE

    python ../utils/ChERRANT/parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char

    nohup python ../utils/ChERRANT/compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE \
        >${OUTPUT_DIR}/yaclc-minimal_dev.cherrant 2>&1 &

done