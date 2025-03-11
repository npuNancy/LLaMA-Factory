# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import re
import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


# 加载模型
print("加载模型 SentenceTransformer: all-MiniLM-L6-v2....")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        self.score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": [],
            "group_num_val": [],
            "group_location_val": [],
            "behaviours_similarity": [],
        }
        return result

    def __post_init__(self):
        self._dump()

    def __call__bakup(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()

    def get_behaviours_similarity(self, behaviours_pred, behaviours_true):
        min_len = min(len(behaviours_pred), len(behaviours_true))

        # 计算 embeddings
        embeddings_pred = sentence_model.encode(behaviours_pred)
        embeddings_true = sentence_model.encode(behaviours_true)

        # 计算成对的相似度 tensor([-0.7437, -0.9973, ……])
        similarities_pairwise = sentence_model.similarity_pairwise(embeddings_pred[:min_len], embeddings_true[:min_len])

        # 取平均
        similarity = similarities_pairwise.mean().item()

        return similarity

    def parse_content_to_operations_and_behaviours(self, content):
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
                # 仅输出步骤 时 operation "Step 1"
                # 完整的 operation "Step 1: Scroll down to see more apps on the device."
                pattern = r"^([sS]tep\s\d+)"
                step = re.findall(pattern, operation)
                if step:
                    step = step[0].strip()
                    operations_dict[step] = operation
                else:
                    print(f"格式不正确:{operation=}")
                    operations_dict[""] = operation

        if operations_dict:  # 如果最后一组操作没有结束, 则添加到列表中
            operations_list.append(operations_dict)

        # behaviours 按行切分
        behaviours_list = [x.strip() for x in behaviours.split("\n") if x.strip() != ""]

        return operations_list, behaviours_list

    def eval_stage_2(self, decoded_preds, decoded_labels):

        # 遍历预测结果和标签
        for pred, label in zip(decoded_preds, decoded_labels):
            # 解析出 操作序列 和 行为序列
            # behaviours_*: ["xxx", "xxx", ……]
            # operations_dict_*: [{"Step 1": "xxx", "Step 2": "xxx"}, {……}， ……]
            operations_dict_true, behaviours_true = self.parse_content_to_operations_and_behaviours(label)
            operations_dict_pred, behaviours_pred = self.parse_content_to_operations_and_behaviours(pred)

            #######################
            # 1. 分组数量是否正确  #
            #######################
            group_num_val = len(operations_dict_true) == len(operations_dict_pred) == len(behaviours_pred)  # 是否相等
            self.score_dict["group_num_val"].append(round(group_num_val * 100, 4))

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
            self.score_dict["group_location_val"].append(round(group_location_val * 100, 4))

            #########################
            # 3. behaviours 是否正确 #
            #########################
            behaviours_similarity = self.get_behaviours_similarity(behaviours_pred, behaviours_true)
            self.score_dict["group_location_val"].append(round(behaviours_similarity, 4))

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        # 将预测结果和标签转换为numpy数组
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # 将预测结果和标签中的IGNORE_INDEX替换为pad_token_id
        # np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        # 将预测结果和标签解码为文本
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算阶段2指标
        self.eval_stage_2(decoded_preds, decoded_labels)

        # 遍历预测结果和标签
        for pred, label in zip(decoded_preds, decoded_labels):
            # 使用jieba分词
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            # 如果预测结果或标签为空，则设置得分为0
            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                # 使用rouge计算得分
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            # 将得分添加到score_dict中
            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            # 使用bleu计算得分
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        # 如果compute_result为True，则返回_dump方法的结果
        if compute_result:
            return self._dump()
