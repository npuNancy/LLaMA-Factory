# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_maml_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # 加载tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # 获取模板并修复tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 获取数据集
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    # 加载模型
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # 创建数据收集器
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # 创建参考模型 ref_model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # 使用相同的模型作为参考模型
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # 设置训练参数 remove_unused_columns, 对于多模态和成对数据集很重要
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

    # Initialize our Trainer 初始化训练器
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training 训练过程
    if training_args.do_train:
        # 训练模型
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # 保存模型
        trainer.save_model()
        # 如果需要计算有效tokens每秒
        if finetuning_args.include_effective_tokens_per_second:
            # 计算有效tokens每秒
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="rm"
            )

        # 记录训练结果
        trainer.log_metrics("train", train_result.metrics)
        # 保存训练结果
        trainer.save_metrics("train", train_result.metrics)
        # 保存训练状态
        trainer.save_state()
        # 如果是主进程并且需要绘制损失图
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            # 绘制损失图
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation 评估过程
    if training_args.do_eval:
        # 评估模型
        metrics = trainer.evaluate(metric_key_prefix="eval")
        # 如果参考模型和当前模型是同一个模型，则无法计算奖励
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            # 找到所有包含"rewards"的键
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            # 删除所有包含"rewards"的键
            for key in remove_keys:
                metrics.pop(key)
        # 记录评估指标
        trainer.log_metrics("eval", metrics)
        # 保存评估指标
        trainer.save_metrics("eval", metrics)

    # Create model card, 创建模型卡片用于展示
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
