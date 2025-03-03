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
    iPersonal_GUI_stage2_aitw_len_10_test
)

datasets=(
    iPersonal_GUI_stage2_android_control_len_1_test
    iPersonal_GUI_stage2_android_control_len_2_test
    iPersonal_GUI_stage2_android_control_len_3_test
    iPersonal_GUI_stage2_android_control_len_4_test
    iPersonal_GUI_stage2_android_control_len_5_test
    iPersonal_GUI_stage2_android_control_len_10_test
)

datasets=(
    iPersonal_GUI_stage2_aitw_len_1_test
    iPersonal_GUI_stage2_aitw_len_2_test
    iPersonal_GUI_stage2_aitw_len_3_test
    iPersonal_GUI_stage2_aitw_len_4_test
    iPersonal_GUI_stage2_aitw_len_5_test
    iPersonal_GUI_stage2_android_control_len_1_test
    iPersonal_GUI_stage2_android_control_len_2_test
    iPersonal_GUI_stage2_android_control_len_3_test
    iPersonal_GUI_stage2_android_control_len_4_test
    iPersonal_GUI_stage2_android_control_len_5_test
)

######## 运行前检查修改 ########
cuda_device="4,5"
checkpoint="4000"
######## 运行前检查修改 ########
adapter_path="/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora/sft"
adapter_path="/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora/sft/checkpoint-$checkpoint"
save_dir="/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora/predict_20250301_cp$checkpoint"


model_path="/data4/yanxiaokai/Models/modelscope/hub/Qwen/Qwen2.5-VL-7B-Instruct"
template="qwen2_vl"

## 创建保存目录
mkdir -p "$save_dir"

# 进入  /data4/yanxiaokai/LLaMA-Factory 目录
cd /data4/yanxiaokai/LLaMA-Factory

# 激活 llama_factory 环境
# conda activate llama_factory

## 遍历数据集
for dataset in "${datasets[@]}"; do
    ## 提取数据集名称中的数字部分
    len=$(echo "$dataset" | grep -oP 'len_\K\d+')

    ## 删除 iPersonal_GUI_stage2_ 
    dataset_name=$(echo "$dataset" | sed 's/iPersonal_GUI_stage2_//g' | sed 's/_test//g' )

    ## 构建保存路径
    save_name="$save_dir/predict_$dataset_name.jsonl"
    

    ## 运行推理命令
    CUDA_VISIBLE_DEVICES="$cuda_device" python scripts/vllm_infer.py \
        --model_name_or_path "$model_path" \
        --adapter_name_or_path "$adapter_path" \
        --template "$template" \
        --cutoff_len 10000 \
        --dataset "$dataset" \
        --save_name "$save_name"



    # 捕获退出状态
    # result=$?

    # if [ $result -eq 0 ]; then
    #     echo "完成 $dataset 的推理，结果保存到 $save_name"
    # else
    #     echo "推理 $dataset 失败，退出状态码 $result"
    #     exit $result
    # fi
    echo "完成 $dataset 的推理，结果保存到 $save_name"
    echo -e "--------------------------------------------------\n\n\n"

    sleep 1
done