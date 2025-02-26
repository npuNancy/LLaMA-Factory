"""
对阶段2的数据进行评估
1. 分组数量是否正确
2. 每组的 low_ins 是否正确(</br>的位置)
3. 每组的 high_ins 是否正确
"""

import os
import json
from typing import List, Dict

from sentence_transformers import SentenceTransformer

# 加载模型
print("加载模型....")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("加载模型完成")


def get_behaviours_similarity(behaviours_pred, behaviours_true):
    min_len = min(len(behaviours_pred), len(behaviours_true))

    # 计算 embeddings
    embeddings_pred = model.encode(behaviours_pred)
    embeddings_true = model.encode(behaviours_true)

    # 计算成对的相似度 tensor([-0.7437, -0.9973, ……])
    similarities_pairwise = model.similarity_pairwise(embeddings_pred[:min_len], embeddings_true[:min_len])

    # 取平均
    similarity = similarities_pairwise.mean().item()

    return similarity


def parse_content_to_operations_and_behaviours(content):
    """
    Description:
        解析content中的操作和动作
    Args:
        content (str): 包含操作和动作的字符串
            *** split_operations ***
            Step 1: Scroll down to see more apps on the device.
            Step 2: Click on the "Play Store" icon to open it.
            </br>
            ……
            *** behaviours ***
            open app "Grab" (install if not already installed) and enter user name: "interspersion@gmail.com" and password: "aristocrats"<|im_end|>
    Returns:
        operations_list (list[Dict]): 包含多组操作的列表
        behaviours_list (List[str]): 包含动作的字符串
    """

    if "*** split_operations ***" not in content:
        return [], []
    if "*** behaviours ***" not in content:
        return [], []

    operations = content.split("*** behaviours ***")[0].replace("*** split_operations ***", "").strip()
    behaviours = content.split("*** behaviours ***")[1].replace("<|im_end|>", "").strip()

    # operations 解析成 [{"Step K": "Scroll down to see more apps on the device."}]
    operations_list = []  # 用于存储多组操作
    operations_dict = {}  # 用于存储单组操作
    for operation in operations.split("\n"):
        operation = operation.strip()
        if operation == "":
            continue
        if operation == "</br>":
            # 分隔符 </br>, 表示一组操作结束, 将当前组操作添加到列表中
            operations_list.append(operations_dict)
            operations_dict = {}  # 清空字典
        else:
            # 正常的 operation
            step = operation.split(":", 1)[0].strip()
            operation_ = operation.split(":", 1)[1].strip()
            operations_dict[step] = operation_

    if operations_dict:  # 如果最后一组操作没有结束, 则添加到列表中
        operations_list.append(operations_dict)

    # behaviours 按行切分
    behaviours_list = [x.strip() for x in behaviours.split("\n") if x.strip() != ""]

    return operations_list, behaviours_list


# 对每一组数据进行评估
def evaluate_prediction(item):
    """
    Description:
        评估预测结果
            1. 分组数量是否正确
            2. 每组的 low_ins 是否正确(</br>的位置)
            3. 每组的 behaviours 是否正确
    Args:
        item (dict): 包含预测结果和真实标签的数据项
            item = {
                "prompt": "",
                "predict": "",
                "label": "",
            }
    Returns:
        dict: 包含评估结果的数据项
    """
    prompt = item["prompt"]
    # prompt 需要包含: <|im_start|>system  <|im_start|>user <|im_start|>assistant
    if not all([x in prompt for x in ["<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant"]]):
        print("prompt 需要包含: <|im_start|>system,  <|im_start|>user,  <|im_start|>assistant")
        return {}

    # behaviours_*: ["xxx", "xxx", ……]
    # operations_dict_*: [{"Step 1": "xxx", "Step 2": "xxx"}, {……}， ……]
    operations_dict_true, behaviours_true = parse_content_to_operations_and_behaviours(item["label"])
    operations_dict_pred, behaviours_pred = parse_content_to_operations_and_behaviours(item["predict"])

    #######################
    # 1. 分组数量是否正确  #
    #######################
    assert len(operations_dict_true) == len(behaviours_true), "GroundTruth 的操作和动作数量不一致"
    group_num_val = len(operations_dict_true) == len(operations_dict_pred) == len(behaviours_pred)  # 是否相等

    #######################
    # 2. </br>位置是否正确 #
    #######################
    if group_num_val == True:
        group_location_sum = 0  # 计算正确率. 但是要是有一个错了，基本上后面的都是错的
        operations_dict_zipped = zip(operations_dict_pred, operations_dict_true)
        for sub_seq_pred, sub_seq_true in operations_dict_zipped:
            # 验证每个子序列的长度是否一致. 一致 <=> </br>的位置正确
            group_location_sum += 1 if len(sub_seq_pred.keys()) == len(sub_seq_true.keys()) else 0
        group_location_val = group_location_sum / len(operations_dict_true)
    else:
        # 如果分组数量不正确, 则不需要判断 </br>的位置
        group_location_val = 0

    #########################
    # 3. behaviours 是否正确 #
    #########################
    behaviours_similarity = get_behaviours_similarity(behaviours_pred, behaviours_true)

    return {
        "group_num_val": group_num_val,
        "group_location_val": group_location_val,
        "behaviours_similarity": behaviours_similarity,
    }


def main(filepath, save_dir):
    # 读取数据
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read().splitlines()

    group_num_val_list = []
    group_location_val_list = []
    behaviours_similarity_list = []

    # 对每一组数据进行评估
    for item in data[:10]:
        prediction = json.loads(item)
        result = evaluate_prediction(prediction)
        group_num_val_list.append(result["group_num_val"])
        group_location_val_list.append(result["group_location_val"])
        behaviours_similarity_list.append(result["behaviours_similarity"])

    # 计算评估结果
    group_num_val = sum(group_num_val_list) / len(group_num_val_list)
    group_location_val = sum(group_location_val_list) / len(group_location_val_list)
    behaviours_similarity = sum(behaviours_similarity_list) / len(behaviours_similarity_list)

    filename = os.path.basename(filepath)
    print(f"{filename=}")
    print(f"分组数量正确率: {group_num_val}")
    print(f"</br>位置正确率: {group_location_val}")
    print(f"behaviours 相似度: {behaviours_similarity}")

    save_file_name = filename.replace(".jsonl", "_eval.csv")
    save_file_path = os.path.join(save_dir, save_file_name)

    save_file_path = save_dir  # 整合到1个csv文件中

    with open(save_file_path, "a", encoding="utf-8") as f:
        # 判断当前指针是否是在文件开头
        if f.tell() == 0:
            f.write(f"文件名, 分组数量正确率, </br>位置正确率, behaviours 相似度\n")
        f.write(f"{filename}, {group_num_val}, {group_location_val}, {behaviours_similarity}\n")


if __name__ == "__main__":
    evaluation_dir = "/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora/evaluation"
    evaluation_dir = "/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora/evaluation/evaluations.csv"
    predict_dir = "/data4/yanxiaokai/LLaMA-Factory/saves/qwen25vl_7B_stage2/lora/predict"

    main(os.path.join(predict_dir, "predict_aitw_len_1.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_aitw_len_2.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_aitw_len_3.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_aitw_len_4.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_aitw_len_5.jsonl"), save_dir=evaluation_dir)

    main(os.path.join(predict_dir, "predict_android_control_len_1.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_android_control_len_2.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_android_control_len_3.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_android_control_len_4.jsonl"), save_dir=evaluation_dir)
    main(os.path.join(predict_dir, "predict_android_control_len_5.jsonl"), save_dir=evaluation_dir)
