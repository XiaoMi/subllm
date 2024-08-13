# !/bin/bash 

python ./stream_inference_rightpad_subllm.py \
    --model_path $path_to_subllm \
    --model_config_path ../conifg/subllm/$model_config_file \
    --tokenizer_path $path_to_tokenizer \
    --data_path $data_for_inference \
    --save_path $result_path \
    --bf16 \
    --use_cache \
    --temperature 0 \
    --repetition_penalty 1.5 \
    --timing \
    --num_avg_times 5 \
    --batch_size 5 \
    --max_length $max_len \
    --max_input_len $max_input_len 
    