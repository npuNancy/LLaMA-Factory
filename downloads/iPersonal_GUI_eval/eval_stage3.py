import os
import re
import json
import numpy as np
from typing import List, Dict

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
    # 对每一组数据进行评估
    for item in data:
        item = json.loads(item)
        predict = json.loads(item["predict"])["Proactive Task"]
        label = json.loads(item["label"])["Proactive Task"]
        predict_list.append(predict)
        label_list.append(label)

        acc.append(int(predict == label))  # 判断是否相等

    # 计算 embeddings
    embeddings_pred = model.encode(predict_list)
    embeddings_true = model.encode(label_list)
    # 计算成对的相似度 tensor([-0.7437])
    similarities_pairwise = model.similarity_pairwise(embeddings_pred, embeddings_true)

    # 计算准确率
    acc = sum(acc) / len(acc) * 100.0
    sim_acc = (similarities_pairwise > threshold).float().mean().item() * 100.0
    print(f"Accuracy: {acc:.2f}%")
    print(f"Similarity Accuracy: {sim_acc:.2f}%")


if __name__ == "__main__":
    dir = "/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage3/lora/predict_cp4750_same_user/generated_predictions.jsonl"
    main(dir)
