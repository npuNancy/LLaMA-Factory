#!/bin/bash
# 批量运行 VLLM 推理，并将不同数据集的结果保存到不同的文件中。
# 在 LLaMA-Factory/ 下运行脚本 ./downloads/iPersonal_GUI_inference/run_vllm_inference.sh

# 定义数据集列表
datasets=(
    iPersonal_GUI_stage2_aitw_len_1_test
    iPersonal_GUI_stage2_aitw_len_2_test
    iPersonal_GUI_stage2_aitw_len_3_test
    iPersonal_GUI_stage2_aitw_len_4_test
    iPersonal_GUI_stage2_aitw_len_5_test
)

datasets=(
    iPersonal_GUI_stage2_android_control_len_1_test
    iPersonal_GUI_stage2_android_control_len_2_test
    iPersonal_GUI_stage2_android_control_len_3_test
    iPersonal_GUI_stage2_android_control_len_4_test
)

## 定义其他固定参数
cuda_device=4
model_path="/data4/yanxiaokai/Models/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct"
adapter_path="/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora/sft/checkpoint-27000"
template="qwen2_vl"

## 创建保存目录
save_dir="/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora"
mkdir -p "$save_dir"

## 遍历数据集
for dataset in "${datasets[@]}"; do
    ## 提取数据集名称中的数字部分
    len=$(echo "$dataset" | grep -oP 'len_\K\d+')
    
    ## 构建保存路径
    save_name="$save_dir/predict_aitw_len_$len.jsonl"
    save_name="$save_dir/predict_android_control_len_$len.jsonl"
    

    ## 运行推理命令
    CUDA_VISIBLE_DEVICES=$cuda_device python scripts/vllm_infer.py \
        --model_name_or_path "$model_path" \
        --adapter_name_or_path "$adapter_path" \
        --template "$template" \
        --cutoff_len 10000 \
        --dataset "$dataset" \
        --save_name "$save_name"

    
    echo "完成 $dataset 的推理，结果保存到 $save_name"
    echo -e "--------------------------------------------------\n\n\n"

    # Sleep 5 秒
    sleep 5
done