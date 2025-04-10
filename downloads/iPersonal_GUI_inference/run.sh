# 每次运行不同的脚本

## 20250407
# echo "sft_100user_20way_event_en"
# CUDA_VISIBLE_DEVICES=0 lmf train \
#     --model_name_or_path /data4/yanxiaokai/Models/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct \
#     --adapter_name_or_path saves/qwen25vl_7B_stage3/lora/sft_100user_20way_event_en/checkpoint-36500  \
#     --output_dir saves/qwen25vl_7B_stage3/lora/predict_100user_20way_event_en \
#     --eval_dataset iPersonal_GUI_stage3_sft_100user_20way_event_test \
#     --stage sft \
#     --do_predict \
#     --finetuning_type lora \
#     --infer_backend huggingface \
#     --new_special_tokens "<Role>,</Role>,<Task>,</Task>,<Format>,</Format>,<Rules>,</Rules>,<choices>,</choices>" \
#     --template qwen2_vl \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 10000 \
#     --max_samples 1000 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --predict_with_generate

## 20250407
# echo "sft_100user_20way_event_zh2en"
# CUDA_VISIBLE_DEVICES=0 lmf train \
#     --model_name_or_path /data4/yanxiaokai/Models/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct \
#     --adapter_name_or_path saves/qwen25vl_7B_stage3/lora/sft_100user_20way_event_zh2en/checkpoint-19000  \
#     --output_dir saves/qwen25vl_7B_stage3/lora/predict_100user_20way_event_zh2en \
#     --eval_dataset iPersonal_GUI_stage3_sft_100user_20way_event_zh2en_test \
#     --stage sft \
#     --do_predict \
#     --finetuning_type lora \
#     --infer_backend huggingface \
#     --new_special_tokens "<Role>,</Role>,<Task>,</Task>,<Format>,</Format>,<Rules>,</Rules>,<choices>,</choices>" \
#     --template qwen2_vl \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 10000 \
#     --max_samples 1000 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --predict_with_generate

## 20250410
API_PORT=8002 CUDA_VISIBLE_DEVICES=0 lmf api \
    --model_name_or_path /data4/yanxiaokai/Models/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_name_or_path saves/qwen25vl_7B_stage3/lora/sft_100user_20way_event_en/checkpoint-36500 \
    --template qwen2_vl \
    --finetuning_type lora \
    --infer_backend vllm \
    --vllm_gpu_util 0.3

echo "完成"