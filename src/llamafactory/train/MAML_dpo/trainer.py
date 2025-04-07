# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        # 如果transformers版本大于4.46，则将kwargs中的tokenizer参数赋值给processing_class参数
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        # 如果disable_dropout为True，则禁用模型中的dropout
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning 破解以避免警告
        self.generate_during_eval = False  # disable at evaluation 评估时禁用
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams 超参数
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        # 调用父类的初始化方法
        Trainer.__init__(self, model=model, **kwargs)
        # 覆盖trainer的默认行为
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        # 删除ref模型上的gc警告
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        # 如果ref_model不为None，则根据是否使用deepspeed来准备ref_model
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        # 如果processor不为None，则添加SaveProcessorCallback回调
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # 如果使用badam，则添加BAdamCallback回调
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        """
        设置优化器。

        我们提供了一个合理且有效的默认设置。
        如果你想使用其他东西，你可以通过`optimizers`在Trainer的init中传递一个元组，或者在子类中重写这个方法。
        """
        # 如果optimizer为None，则创建自定义的optimizer
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        # 创建自定义的scheduler
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        # 如果disable_shuffling为True，则使用SequentialSampler
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def get_batch_samples(self, epoch_iterator, num_batches):
        r"""
        Replaces the method of KTO Trainer with the one of the standard Trainer.
        将 KTO Trainer 的方法替换为标准 Trainer 的方法。
        """
        return Trainer.get_batch_samples(self, epoch_iterator, num_batches)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        计算策略模型的分批对数概率的ORPO比值比（OR）损失。
        """
        # 计算对数概率的比值
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        # 计算Softmax损失
        sft_loss = -chosen_logps
        # 计算比值比损失
        odds_ratio_loss = -F.logsigmoid(log_odds)
        # 计算ORPO损失
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        计算策略模型的批处理日志概率的SimPO损失。
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        计算偏好学习的损失。
        """
        # 如果不使用ref_model，则根据loss_type计算loss
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            # 如果使用ref_model，则调用dpo_loss计算loss
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.
        如果loss_type不是IPO、ORPO或SimPO，则计算给定logits下标签的对数概率总和。

        Otherwise the average log probabilities.
        否则为平均对数概率。
        """
        # 如果使用ref_model，则调用nested_detach方法
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        # 调用模型的forward方法，获取logits
        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        # 调用get_batch_logps方法，获取logps和valid_length
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        # 如果loss_type是IPO、ORPO或SimPO，则将logps除以valid_length
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        # 将logps和logits分别分为chosen和rejected两部分
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        # 如果loss_type是IPO、ORPO或SimPO，则返回chosen_logps、rejected_logps、chosen_logits、rejected_logits和chosen_logps
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            # 否则返回chosen_logps、rejected_logps、chosen_logits、rejected_logits和chosen_logps/valid_length
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        计算参考模型的对数概率。
        """
        # 如果不使用ref_model，则返回None
        if not self.finetuning_args.use_ref_model:
            return None, None

        # 如果ref_model为None，则将model赋值给ref_model，并调用disable_adapter方法
        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            # 否则将ref_model赋值给ref_model，并创建nullcontext
            ref_model = self.ref_model
            ref_context = nullcontext()

        # 在no_grad和ref_context下，调用concatenated_forward方法，获取reference_chosen_logps和reference_rejected_logps
        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        计算给定批次的训练或测试输入的DPO损失和其他指标。
        """
        metrics = {}
        # 调用concatenated_forward方法，获取policy_chosen_logps、policy_rejected_logps、policy_chosen_logits、policy_rejected_logits和policy_chosen_logps_avg
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        # 调用compute_reference_log_probs方法，获取reference_chosen_logps和reference_rejected_logps
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        # 调用compute_preference_loss方法，获取losses、chosen_rewards和rejected_rewards
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        # 计算sft_loss
        sft_loss = -policy_chosen_logps_avg
        # 如果ftx_gamma大于1e-6，则将sft_loss加到losses中
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        # 根据train_eval的值，设置prefix
        prefix = "eval_" if train_eval == "eval" else ""
        # 计算并添加metrics
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: Dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
        r"""
        Subclass and override to accept extra kwargs.
        子类和重写以接受额外的kwargs。
        """
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        r"""
        Log `logs` on the various objects watching training, including stored metrics.
        记录观看训练的各种对象的“日志”，包括存储的指标。
        """
        # logs either has "loss" or "eval_loss" 日志要么有“loss”，要么有“evalloss”
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs 将平均存储指标添加到日志中
        key_list, metric_list = [], []
        # 遍历self._stored_metrics[train_eval]中的每个key和metrics
        for key, metrics in self._stored_metrics[train_eval].items():
            # 将key添加到key_list中
            key_list.append(key)
            # 将metrics转换为torch.tensor类型，并移动到accelerator.device上，然后计算平均值，并将结果转换为浮点数，最后将结果添加到metric_list中
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        # 删除self._stored_metrics[train_eval]中的所有元素
        del self._stored_metrics[train_eval]
        # 如果metric_list的长度小于10，则进行填充
        if len(metric_list) < 10:  # pad to for all reduce
            # 遍历需要填充的次数
            for i in range(10 - len(metric_list)):
                # 将"dummy_" + i添加到key_list中
                key_list.append(f"dummy_{i}")
                # 将0.0添加到metric_list中
                metric_list.append(0.0)

        # 将metric_list转换为torch.tensor类型，并移动到accelerator.device上
        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        # 使用accelerator.reduce函数对metric_list进行平均操作，并将结果转换为列表
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        # 遍历key_list和metric_list中的每个元素
        for key, metric in zip(key_list, metric_list):  # add remaining items
            # 如果key不以"dummy_"开头，则将key和metric添加到logs中
            if not key.startswith("dummy_"):
                logs[key] = metric

        # 返回Trainer.log函数的结果
        return Trainer.log(self, logs, *args, **kwargs)
