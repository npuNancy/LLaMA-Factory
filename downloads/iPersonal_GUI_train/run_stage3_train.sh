#!/bin/bash
# 使用脚本进行 第三阶段 训练
# FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=3,4,5 lmf train downloads/iPersonal_GUI_train/qwen25vl_7B_stage3.yaml  > /data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage3/lora/train_log_20250303.txt 2>&1

############## 运行前检查并修改 ##########
force_torchrun=0    # 是否强制使用 torchrun
cuda_device="0"   # 使用的 GPU 设备
output_dir="saves/qwen25vl_7B_stage3/lora/sft" # checkpoint 保存目录
############## 运行前检查并修改 ##########

# 模型训练的配置文件路径
train_config="/data4/yanxiaokai/LLaMA-Factory/downloads/iPersonal_GUI_train/qwen25vl_7B_stage3.yaml"


# 修改配置文件中的 output_dir 字段
sed -i "/^output_dir:/s|.*|output_dir: $output_dir|" $train_config

## 日志
# 保存训练日志的目录
save_dir="/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage3/lora/log"
# 动态生成一个带有时间戳的日志文件名
log_file="${save_dir}/train_log_$(date +%Y%m%d).txt"

# 检查 save_dir 是否存在，如果不存在则创建
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
    echo "创建目录: $save_dir"
fi

# 进入  /data4/yanxiaokai/LLaMA-Factory 目录
cd /data4/yanxiaokai/LLaMA-Factory

# 训练
FORCE_TORCHRUN=${force_torchrun} CUDA_VISIBLE_DEVICES="$cuda_device" lmf train ${train_config} > ${log_file} 2>&1

# 捕获 lmf 的退出状态
result=$?

if [ $result -eq 0 ]; then
    echo "训练任务成功完成！"
else
    echo "训练任务失败，退出代码: $result"
    exit $result
fi