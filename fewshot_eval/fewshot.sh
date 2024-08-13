#!/bin/bash -l

MODEL_PATH=$1
CONFIG_PATH=$2
TOKENIZER_PATH=$3
RSLT_PATH=$4
MAX_LEN=$5
TASK=$6
N_SHOT=$7

# evaluate on popular benchmarks
python ./fewshot.py \
    --model_path $MODEL_PATH \
    --config_path $CONFIG_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --task=$TASK \
    --n_shot=$N_SHOT \
    --max_length $MAX_LEN \
    --seed=12345 \
    --device=0 \
    --batch_size=1 \
    --result_path $RSLT_PATH &


# evaluate on other datasets
for seed in {42..57}; do
  echo "$((seed % 8))"
  python ./fewshot.py \
    --model_path $MODEL_PATH \
    --config_path $CONFIG_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --task=$TASK \
    --n_shot=$N_SHOT \
    --max_length $MAX_LEN \
    --seed="${seed}" \
    --device="$((seed % 8))" \
    --batch_size=1 \
    --result_path $RSLT_PATH &
done
wait
