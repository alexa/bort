#!/bin/bash
set -x
set -v

#pkill -f 'python run_pretraining_distillation_hvd'

NUMPROC=$1
TRAIN_DATA_DIR=$2
TEST_DATA_DIR=$3
TEACHER_CKPT=$4
if [ ! -z $TEACHER_CKPT ]
then
    TEACHER_CKPT="--teacher_ckpt_dir=$TEACHER_CKPT"
fi

CPU_ONLY=""
DTYPE="float16"
NUM_STEPS=2000000
LR=1e-4
BS=1024
BS_EVAL=16
ACC=1
LOG_INTERVAL=10
CKPT_INTERVAL=10

MODEL_NAME=bort_4_8_768_1024
DIR_NAME=${MODEL_NAME}_model

if [ "$TRAIN_DATA_DIR" = "TEST" ]
then
    DTYPE="float32"
    CPU_ONLY=" --cpu_only "
    TEST_DIR="pretraining_distillation_test"
    rm -rf $TEST_DIR
    TEST_TXT="test_data/sample_text.txt"
    if [ ! -d $TEST_DIR ]
    then
        mkdir -p $TEST_DIR
        python create_pretraining_data.py \
            --input_file=$TEST_TXT \
            --output_dir=$TEST_DIR \
            --dataset_name openwebtext_ccnews_stories_books_cased --dupe_factor 1 --num_workers 1
        TOKENIZED=$TEST_DIR/part-000.npz
        i=1
        while (($i < $NUMPROC))
        do
            cp $TOKENIZED $TEST_DIR/part-00${i}.npz
            i=$((i+1))
        done
    fi

    DIR_NAME=${TEST_DIR}/${MODEL_NAME}_model
    TRAIN_DATA_DIR="$TEST_DIR/*.npz"
    TEST_DATA_DIR="$TEST_DIR/*.npz"
    NUM_STEPS=4
    BS=2
    BS_EVAL=2
    LOG_INTERVAL=2
    CKPT_INTERVAL=2
fi

# Run training
mkdir -p $DIR_NAME
touch ${DIR_NAME}/train.log

export MXNET_SAFE_ACCUMULATION=1
PER_HOST_GPU_CNT=8
horovodrun -np $NUMPROC -H localhost:$NUMPROC \
         python run_pretraining_distillation_hvd.py \
         --dataset_name openwebtext_ccnews_stories_books_cased \
         --data=$TRAIN_DATA_DIR \
         --data_eval=$TEST_DATA_DIR \
         --teacher_model 'roberta_24_1024_16' \
         $TEACHER_CKPT \
         --teacher_ce_weight 0.5 \
         --distillation_temperature 2.0 \
         --mlm_weight 0.1 \
         --num_steps $NUM_STEPS \
         --lr $LR \
         --batch_size_eval $BS_EVAL \
         --batch_size $BS \
         --accumulate $ACC \
         --num_buckets 20 \
         --use_avg_len \
         --log_interval $LOG_INTERVAL \
         --ckpt_interval $CKPT_INTERVAL \
         --model $MODEL_NAME \
         --ckpt_dir $DIR_NAME \
         --dtype $DTYPE \
         $CPU_ONLY \
         &> ${DIR_NAME}/train.log
py_result=$?
set +v
set +x
exit ${py_result}