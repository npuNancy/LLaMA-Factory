import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import json
import numpy as np
from typing import List, Dict
from json.decoder import JSONDecodeError

from sentence_transformers import SentenceTransformer

from transformers.utils import is_jieba_available, is_nltk_available

if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# 加载模型
print("加载模型....")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("加载模型完成")


def main(filepath, threshold=0.95):
    # 读取数据
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read().splitlines()

    acc = []
    predict_list = []
    label_list = []
    category_dict = {}  # 每一个类别的准确率
    # 对每一组数据进行评估
    for item in data:
        try:
            item = item.replace("<|im_start|>", "").replace("<|im_end|>", "")
            if "Proactive Task" in item:  # 如果包含 "Proactive Task" 字段
                item = json.loads(item)
                predict = json.loads(item["predict"].strip())["Proactive Task"]
                label = json.loads(item["label"].strip())["Proactive Task"]
            else:
                item = json.loads(item)
                predict = item["predict"].strip()
                label = item["label"].strip()
        except JSONDecodeError:
            predict = item["predict"].strip()
            label = item["label"].strip()
        except:
            predict = "null"
            label = "null"

        predict_list.append(predict)
        label_list.append(label)

        acc.append(int(predict == label))  # 判断是否相等

        # 统计每个类别的准确率
        if label not in category_dict:
            category_dict[label] = []
        category_dict[label].append(int(predict == label))

    # 计算 embeddings
    embeddings_pred = model.encode(predict_list)
    embeddings_true = model.encode(label_list)
    # 计算成对的相似度 tensor([-0.7437])
    similarities_pairwise = model.similarity_pairwise(embeddings_pred, embeddings_true)

    # 计算准确率
    accuracy = sum(acc) / len(acc) * 100.0
    sim_acc = (similarities_pairwise > threshold).float().mean().item() * 100.0
    print(f"Acc: {accuracy:.2f}%")
    print(f"SimAcc: {sim_acc:.2f}%")

    # 输出每个类别的准确率
    category_acc_dict = {category: sum(value) / len(value) * 100.0 for category, value in category_dict.items()}
    # 计算平均类别准确率
    avg_category_acc = np.mean(list(category_acc_dict.values()))
    print(f"AvgAcc: {avg_category_acc:.2f}%")


if __name__ == "__main__":
    filepath = "saves/qwen25_7B_stage3/lora/predict_100user_20way_event/predict_100user_20way_event.jsonl"  # Acc: 39.94% SimAcc: 44.54% AvgAcc: 27.80%

    filepath = "saves/qwen25vl_7B_stage3/lora/predict_cp4750_same_user/generated_predictions.jsonl"  # 22.6%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_20way_sft_ours/predict_20way.jsonl"  # 34.94%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_20way_sft_qwen2.5VL7B/predict_20way_qwen2.5VL7B.jsonl"  # 18.72%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_20way_sft_gpt-3.5-turbo/generated_predictions.jsonl"  # 7.35%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_20way_sft_gpt-4o/generated_predictions.jsonl"  # 7.35%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_100user_200way_event/generated_predictions.jsonl"  # Acc: 31.2%, AvgAcc: 18.72%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_100user_20way_event_zh/generated_predictions.jsonl"  # Acc: 45.00% SimAcc: 46.90% AvgAcc: 31.48%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_100user_20way_event_zh2en/generated_predictions.jsonl"  # Acc: 39.10%, SimAcc: 39.10%, AvgAcc: 26.72%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_100user_20way_event(cp_15000)/generated_predictions.jsonl"  # Acc: 35.80%, SimAcc: 35.80%, AvgAcc: 24.39%
    filepath = "saves/qwen25vl_7B_stage3/lora/predict_100user_20way_event_en/generated_predictions.jsonl"  # Acc: 42.20%, SimAcc: 42.20%, AvgAcc: 33.83%
    main(filepath=filepath)
