#!/bin/bash
TASK=$1

# Default parameters. Refer to the paper for their values.
NO_DISTRIBUTED=""
EPS=1
REPS=1
BSZ=8
SEED=2
LR=5e-5
MAXLEN=512
WD=110.0
WMUP=0.45
GPU=0

# Test GLUE
if [ "$TASK" = "TEST1" ]; then
  TASK="WNLI"
  EPS=1
  REPS=1
  BSZ=8
  GPU=999
  NO_DISTRIBUTED=" --no_distributed True"
fi

# Test SuperGLUE
if [ "$TASK" = "TEST2" ]; then
  TASK="WSC"
  EPS=1
  REPS=1
  BSZ=8
  GPU=999
  NO_DISTRIBUTED=" --no_distributed True"
fi


python finetune_bort.py \
  --task_name ${TASK} \
  --dataset openwebtext_ccnews_stories_books_cased \
  --pretrained_parameters model/0044100.params \
  --batch_size $BSZ \
  --max_len $MAXLEN \
  --ramp_up_epochs $REPS \
  --epochs $EPS \
  --seed $SEED \
  --gpu $GPU \
  --warmup_ratio $WMUP \
  --weight_decay $WD \
  --early_stop 200 \
  --lr $LR \
  $NO_DISTRIBUTED
