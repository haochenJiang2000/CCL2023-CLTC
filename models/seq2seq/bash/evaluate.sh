source /public/home/swfeng/anaconda3/bin/activate
conda activate cherrant
MODEL_DIR=../model/finetune-i/lang8_v2/seed_2023
OUTPUT_DIR=$MODEL_DIR/results
HYP_PARA_FILE=$OUTPUT_DIR/yaclc-minimal_dev.out.para
HYP_M2_FILE=$OUTPUT_DIR/yaclc-minimal_dev.out.m2
DATA_DIR=../data/yaclc-minimal_dev
REF_M2_FILE=$DATA_DIR/yaclc-minimal_dev.m2

paste $DATA_DIR/single/src.txt $OUTPUT_DIR/yaclc-minimal_dev.out.post_processed | awk '{print NR"\t"$p}' > $HYP_PARA_FILE

python ../utils/ChERRANT/parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char

python ../utils/ChERRANT/compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE