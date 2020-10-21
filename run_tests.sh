#!/bin/bash

rm -f log_test_glue log_test_qa log_test_pretraining

# Test GLUE/SuperGLUE
./run_finetune.sh TEST1 > log_test_glue
glue_result=$?
./run_finetune.sh TEST2 > log_test_qa
superglue_result=$?
# Test pretraining
./run_pretraining_distillation.sh 1 TEST > log_test_pretraining
pretraining_result=$?

if [[ $glue_result -eq 0 ]]; then
    echo "[ OK ]	Classification and GLUE"
else
    echo "[FAIL]	Classification and GLUE"
fi
if [[ $superglue_result -eq 0 ]]; then
    echo "[ OK ]	SuperGLUE"
else
    echo "[FAIL]	SuperGLUE"
fi
if [[ $pretraining_result -eq 0 ]]; then
    echo "[ OK ]	Pretraining"
else
    echo "[FAIL]	Pretraining"
fi
