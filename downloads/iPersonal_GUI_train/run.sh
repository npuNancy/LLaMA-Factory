## 20250407
## sft_100user_20way_event_en/checkpoint-36500 断点重训
FORCE_TORCHRUN=0 CUDA_VISIBLE_DEVICES=2 lmf train \
    --model_name_or_path /data4/yanxiaokai/Models/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct \
    --resume_from_checkpoint saves/qwen25vl_7B_stage3/lora/sft_100user_20way_event_en/checkpoint-36500 \
    --output_dir saves/qwen25vl_7B_stage3/lora/sft_100user_20way_event_en \
    --dataset iPersonal_GUI_stage3_sft_100user_20way_event_train \
    --num_train_epochs 3 \
    --stage sft \
    --do_train true \
    --finetuning_type lora \
    --new_special_tokens "<Role>,</Role>,<Task>,</Task>,<Format>,</Format>,<Rules>,</Rules>,<choices>,</choices>" \
    --template qwen2_vl \
    --cutoff_len 10000 \
    --max_samples 1000000 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --bf16 true \
    --logging_steps 100 \
    --plot_loss true \
    --overwrite_output_dir true \
    --val_size 0.05 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 500

echo "[$(date)] 完成"