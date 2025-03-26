API_PORT=8001 CUDA_VISIBLE_DEVICES=6 lmf api \
    --model_name_or_path /data4/yanxiaokai/Models/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_name_or_path saves/qwen25vl_7B_stage3/lora/sft_100user_20way_event\
    --template qwen2_vl \
    --finetuning_type lora