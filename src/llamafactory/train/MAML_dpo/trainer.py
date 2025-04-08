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

import copy
from torch import nn, optim
from torch.utils.data import Dataset
from peft import PeftModel, PeftConfig

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers import TrainerCallback

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
        # 如果 use_ref_model==False, 则根据loss_type计算loss
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
            # 如果使用 use_ref_model==True, 则调用dpo_loss计算loss
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

        # 如果ref_model为None，则将model赋值给ref_model，并调用 disable_adapter 方法
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


class MAMLDPOTrainer(CustomDPOTrainer):
    """
    基于CustomDPOTrainer实现的MAMLDPOTrainer类
    """


'''
class MAMLDPOTrainerKimi(DPOTrainer):
    """
    A trainer class that combines Model-Agnostic Meta-Learning (MAML) with Direct Preference Optimization (DPO).
    This trainer implements the MAML inner and outer loops on top of the DPO training framework.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args,
        data_collator=None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedModel] = None,
        model_wrapped: Optional[PreTrainedModel] = None,
        reward_model: Optional[PreTrainedModel] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        alpha: float = 0.1,  # Inner loop learning rate
        beta: float = 0.01,  # Outer loop learning rate
        num_inner_steps: int = 5,  # Number of inner loop steps
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_wrapped=model_wrapped,
            reward_model=reward_model,
            callbacks=callbacks,
            **kwargs,
        )

        self.alpha = alpha
        self.beta = beta
        self.num_inner_steps = num_inner_steps

        # Create a temporary model for inner loop updates
        self.inner_model = self._create_inner_model()

    def _create_inner_model(self):
        """Create a model for inner loop updates"""
        # If using PEFT, create a PeftModel for the inner loop
        if isinstance(self.model, PeftModel):
            return PeftModel.from_pretrained(self.model.base_model, self.model.peft_config)
        else:
            return self.model.__class__.from_pretrained(self.model.config)

    def inner_loop_update(self, task_data):
        """Perform inner loop updates for a specific task"""
        # Save original parameters
        original_params = {n: p.clone() for n, p in self.inner_model.named_parameters()}

        # Perform multiple inner loop updates
        for _ in range(self.num_inner_steps):
            # Forward pass
            outputs = self.inner_model(**task_data)

            # Compute loss (using DPO loss)
            loss = self.dpo_loss(outputs.chosen, outputs.rejected)

            # Backward pass
            loss.backward()

            # Update inner model parameters
            with torch.no_grad():
                for n, p in self.inner_model.named_parameters():
                    if p.grad is not None:
                        p -= self.alpha * p.grad
                        p.grad.zero_()

        return original_params

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs=False,
    ) -> torch.Tensor:
        """
        Compute the loss using MAML's outer loop update
        """
        # Split inputs into tasks
        tasks = self._split_into_tasks(inputs)

        # Initialize meta-loss
        meta_loss = 0

        # Outer loop: Iterate over tasks
        for task in tasks:
            # Inner loop: Adapt to task
            original_params = self.inner_loop_update(task)

            # Compute loss on adapted parameters
            outputs = self.inner_model(**task)
            task_loss = self.dpo_loss(outputs.chosen, outputs.rejected)

            # Accumulate meta-loss
            meta_loss += task_loss

            # Restore original parameters for next task
            with torch.no_grad():
                for n, p in self.inner_model.named_parameters():
                    p.copy_(original_params[n])

        # Compute gradients for outer loop update
        meta_loss.backward()

        # Update outer model parameters
        with torch.no_grad():
            for n, p_outer in model.named_parameters():
                if p_outer.grad is not None:
                    p_outer -= self.beta * p_outer.grad
                    p_outer.grad.zero_()

        return meta_loss

    def _split_into_tasks(self, inputs):
        """Split batch into individual tasks for MAML"""
        # This is a placeholder - actual implementation depends on your data structure
        # For demonstration, we assume each batch contains multiple tasks
        # You would need to implement this based on your specific use case

        # Example: Split batch into N tasks
        batch_size = inputs["input_ids"].shape[0]
        num_tasks = batch_size // 2  # Assuming each task has 2 samples

        tasks = []
        for i in range(num_tasks):
            task_data = {
                "input_ids": inputs["input_ids"][i * 2 : (i + 1) * 2],
                "attention_mask": inputs["attention_mask"][i * 2 : (i + 1) * 2],
                # Add other inputs as needed
            }
            tasks.append(task_data)

        return tasks

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a training step with MAML's outer loop update
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass and compute loss
        loss = self.compute_loss(model, inputs)

        return loss.detach()

'''


'''
class MAMLDPOTrainerGPT(DPOTrainer):
    """
    基于 DPOTrainer，添加 MAML 双重循环机制。
    注意：此示例代码仅提供基本架构思路，根据实际任务和 DPOTrainer 内部实现细节可能需要修改。
    """

    def __init__(self, *args, inner_lr: float = 1e-3, inner_steps: int = 1, **kwargs):
        """
        Args:
            inner_lr: 内循环的学习率
            inner_steps: 内循环的梯度更新步数
            *args, **kwargs: 传递给 DPOTrainer 的其他参数
        """
        super().__init__(*args, **kwargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        # 假设训练采用的是 self.model 和 self.optimizer，
        # 这里可以根据需要初始化更多内容，如损失函数
        self.loss_fn = nn.CrossEntropyLoss()  # 示例损失函数，根据实际任务替换

    def compute_loss(self, outputs, targets):
        """
        根据输出和目标计算损失
        这里以交叉熵损失为示例，请根据具体任务修改计算过程。
        """
        return self.loss_fn(outputs, targets)

    def meta_train_step(self, support_batch, query_batch):
        """
        执行一次 meta-training 步骤
        Args:
            support_batch: 支持集，通常包含 (input, target)
            query_batch: 查询集，通常包含 (input, target)
        """

        # 保存当前模型参数（元参数）作为备份
        meta_params = copy.deepcopy(self.model.state_dict())

        # --- 内循环：在支持集上做几步梯度更新 ---
        # 克隆一份模型供内循环使用，防止直接改动 self.model 参数（也可以直接修改 self.model 但后续必须还原）
        # 注意：此处为了简单，直接对 self.model 参数做更新，要求后续将其还原！
        # 如果模型较大、内循环步数多，建议构造专门的辅助模型，用于计算内循环更新后的梯度
        for _ in range(self.inner_steps):
            # 从支持集 batch 中获取数据
            inputs_support, targets_support = support_batch["inputs"], support_batch["targets"]
            outputs_support = self.model(inputs_support)
            support_loss = self.compute_loss(outputs_support, targets_support)
            # 计算梯度，要求创建梯度图以支持外层梯度传播
            grads = torch.autograd.grad(support_loss, self.model.parameters(), create_graph=True)

            # 用内循环学习率更新当前模型参数（这里以简单 SGD 风格更新）
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), grads):
                    param.sub_(self.inner_lr * grad)

        # --- 外循环：在查询集上计算损失，并进行反向传播 ---
        inputs_query, targets_query = query_batch["inputs"], query_batch["targets"]
        outputs_query = self.model(inputs_query)
        query_loss = self.compute_loss(outputs_query, targets_query)

        # 反向传播：query_loss 的梯度将会传回内循环中计算更新时的梯度图
        query_loss.backward()

        # 用原始模型参数（元参数）还原 self.model，确保参数更新是通过优化器进行
        self.model.load_state_dict(meta_params)

        # 最后一步：优化器更新元参数
        self.optimizer.step()
        self.optimizer.zero_grad()

        return query_loss.item()

    def train(self, train_dataloader, meta_iterations: int = 1000):
        """
        重写训练函数，采用 meta-training 框架对数据做迭代更新
        这里假设 train_dataloader 内部每个 batch 已经划分好支持集与查询集（例如，通过字典包含 'support' 和 'query'）
        """
        self.model.train()
        for iteration in range(meta_iterations):
            # 从 dataloader 中获取一个 meta-batch，要求每个 batch 包含 support 与 query 两部分
            try:
                batch = next(iter(train_dataloader))
            except StopIteration:
                # 若 dataloader 用尽，则重新获取
                train_dataloader = iter(train_dataloader)
                batch = next(train_dataloader)

            support_batch = batch["support"]
            query_batch = batch["query"]

            loss = self.meta_train_step(support_batch, query_batch)
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Query Loss: {loss:.4f}")
'''
