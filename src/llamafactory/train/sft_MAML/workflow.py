# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

import copy
from typing import TYPE_CHECKING, List, Dict, Literal, Optional, Sequence, Union

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer, MAMLSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from transformers import PreTrainedTokenizer, ProcessorMixin
    import datasets
    from datasets import Dataset, IterableDataset

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

    from ...data.template import Template
    from ...data.data_utils import DatasetModule

logger = get_logger(__name__)


def get_maml_dataset_list(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
):
    """
    Description:
        获取用于元学习的训练集（后续可以从中抽取支持集和查询集）
    Returns:
        maml_training_dataset_list: List[Union[Dataset, IterableDataset, "datasets.Dataset"]]
                                    用于训练的数据集, 包含不同Task(User)的数据集. 后续可以从中抽取支持集和查询集
        maml_testing_dataset_list: List[Union[Dataset, IterableDataset, "datasets.Dataset"]]
                                    用于测试的数据集, 包含不同Task(User)的数据集. 后续可以从中抽取测试集
    """
    maml_training_dataset_list = []
    maml_testing_dataset_list = []

    # 拷贝一份 data_args
    data_args_copy = copy.deepcopy(data_args)
    for dataset_name in list(data_args.dataset):
        """
        获取数据集
        dataset_module = {
            "train_dataset": Optional[Union["Dataset", "IterableDataset"]]
            "eval_dataset": Optional[Union["Dataset", "IterableDataset"]]
        }
        """
        logger.info_rank0(f"函数: get_maml_dataset_list, 正在加载 {dataset_name=}")
        data_args_copy.dataset = [dataset_name]
        dataset_module = get_dataset(
            template, model_args, data_args_copy, training_args, stage="sft", tokenizer=tokenizer, processor=processor
        )

        if "train_dataset" not in dataset_module.keys():
            logger.info_rank0(f"{dataset_name=}, \t dataset_module 中没有 `train_dataset`")
        else:
            maml_training_dataset_list.append(dataset_module["train_dataset"])
        if "eval_dataset" not in dataset_module.keys():
            logger.info_rank0(f"{dataset_name=}, \t dataset_module 中没有 `eval_dataset`")
        else:
            maml_testing_dataset_list.append(dataset_module["eval_dataset"])

        logger.info_rank0(f"{dataset_name=}, \t 加载完成")
    return maml_training_dataset_list, maml_testing_dataset_list


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    # 加载tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    # 获取模板并修复tokenizer
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 获取元学习数据集
    maml_training_dataset_list, maml_testing_dataset_list = get_maml_dataset_list(
        template, model_args, data_args, training_args, stage="sft", **tokenizer_module
    )

    # 获取数据集 dataset_module={"train_dataset": ..., "eval_dataset": ...}
    # dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    dataset_module = {
        "train_dataset": maml_training_dataset_list[0] if len(maml_training_dataset_list) > 0 else None,
        "eval_dataset": maml_testing_dataset_list[0] if len(maml_testing_dataset_list) > 0 else None,
    }

    # 加载模型
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # 如果模型是量化模型且不进行训练，则进行hack操作，使模型兼容预测
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    # 创建数据收集器
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    # 覆盖Seq2SeqTrainer的解码参数
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset 对多模态数据集很重要

    # Metric utils
    # 指标计算工具
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    # `model.generate`的关键字参数
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Initialize our Trainer
    # 初始化我们的Trainer
    """
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    """

    trainer = MAMLSeq2SeqTrainer(
        maml_training_dataset_list=maml_training_dataset_list,
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    # 训练
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    # 评估
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    # 预测
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    # 创建模型卡片
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
