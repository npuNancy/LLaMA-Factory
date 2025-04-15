# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import os
import json
import time
import collections
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

if True:
    import warnings

    import huggingface_hub.utils as hf_hub_utils
    from huggingface_hub import ModelCard, create_repo, upload_folder

    if TYPE_CHECKING:
        import optuna

    from transformers.integrations import get_reporting_integration_callbacks
    from transformers.configuration_utils import PretrainedConfig
    from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
    from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
    from transformers.image_processing_utils import BaseImageProcessor
    from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
    from transformers.integrations.tpu import tpu_spmd_dataloader
    from transformers.modelcard import TrainingSummary
    from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
    from transformers.optimization import Adafactor, get_scheduler
    from transformers.processing_utils import ProcessorMixin
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_2_3
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import (
        CallbackHandler,
        DefaultFlowCallback,
        ExportableState,
        PrinterCallback,
        ProgressCallback,
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from transformers.trainer_pt_utils import (
        DistributedTensorGatherer,
        EvalLoopContainer,
        IterableDatasetShard,
        LabelSmoother,
        LayerWiseDummyOptimizer,
        LengthGroupedSampler,
        SequentialDistributedSampler,
        distributed_broadcast_scalars,
        distributed_concat,
        find_batch_size,
        get_model_param_count,
        get_module_class_from_name,
        get_parameter_names,
        nested_concat,
        nested_detach,
        nested_numpify,
        nested_xla_mesh_reduce,
        reissue_pt_warnings,
        remove_dummy_checkpoint,
        set_rng_state_for_device,
    )
    from transformers.trainer_utils import (
        PREFIX_CHECKPOINT_DIR,
        BestRun,
        EvalLoopOutput,
        EvalPrediction,
        HPSearchBackend,
        HubStrategy,
        PredictionOutput,
        RemoveColumnsCollator,
        SaveStrategy,
        TrainerMemoryTracker,
        TrainOutput,
        check_target_module_exists,
        default_compute_objective,
        denumpify_detensorize,
        enable_full_determinism,
        find_executable_batch_size,
        get_last_checkpoint,
        has_length,
        neftune_post_forward_hook,
        number_of_arguments,
        seed_worker,
        set_seed,
        speed_metrics,
    )
    from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
    from transformers.utils import (
        ADAPTER_CONFIG_NAME,
        ADAPTER_SAFE_WEIGHTS_NAME,
        ADAPTER_WEIGHTS_NAME,
        CONFIG_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
        WEIGHTS_INDEX_NAME,
        WEIGHTS_NAME,
        XLA_FSDPV2_MIN_VERSION,
        PushInProgress,
        PushToHubMixin,
        can_return_loss,
        find_labels,
        is_accelerate_available,
        is_apex_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_galore_torch_available,
        is_grokadamw_available,
        is_in_notebook,
        is_ipex_available,
        is_liger_kernel_available,
        is_lomo_available,
        is_peft_available,
        is_safetensors_available,
        is_sagemaker_dp_enabled,
        is_sagemaker_mp_enabled,
        is_schedulefree_available,
        is_torch_compile_available,
        is_torch_mlu_available,
        is_torch_mps_available,
        is_torch_musa_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_xla_available,
        is_torch_xpu_available,
        is_torchao_available,
        logging,
        strtobool,
    )
    from transformers.utils.deprecation import deprecate_kwarg
    from transformers.utils.quantization_config import QuantizationMethod


if True:
    from transformers.trainer import *
    from transformers.trainer import _is_peft_model


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    继承Seq2SeqTrainer来计算生成指标，如BLEU和ROUGE。
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    ## 获取 训练数据的 采样器
    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        # 如果禁用了打乱，则返回一个顺序采样器
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        # 否则，调用父类的_get_train_sampler方法
        # 实际上就是 return RandomSampler(self.train_dataset)
        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            # 生成时不将标签传递给模型
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


class MAMLSeq2SeqTrainer(CustomSeq2SeqTrainer):
    """
    Description:
        基于 CustomSeq2SeqTrainer 修改的元学习 Trainer
    Args:
        ------Trainer的参数-------
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    """

    def __init__(
        self,
        num_shots: int,
        num_querys: int,
        maml_training_dataset_list: List[Union[Dataset, IterableDataset, "datasets.Dataset"]],
        maml_inner_epochs: int = 1,
        *args,
        **kwargs,
    ):
        """
        Args:
            num_shots: k-shots
            num_querys: q-querys
            maml_training_dataset_list: List[Union[Dataset, IterableDataset, "datasets.Dataset"]]
        """
        super().__init__(*args, **kwargs)

        self.num_shots = num_shots  # k-shots
        self.num_querys = num_querys  # q-querys
        self.maml_training_dataset_list = maml_training_dataset_list
        self.maml_num_tasks = len(maml_training_dataset_list)  # MAML 训练任务的数量

        # 这个无法使用
        self.maml_inner_epochs = maml_inner_epochs  # MAML 每个任务的训练轮次

    def __get_gpu_memory(self, text: str = None):
        if text:
            logger.info(f"[===GPU MEMORY===] {text} <=========")
        # 查看当前已分配的显存 (单位: Byte)
        allocated = torch.cuda.memory_allocated(device=None)  # 默认当前设备
        logger.info(f"[===GPU MEMORY===] 已分配显存: {allocated / 1024**2:.2f} MB <=========")

        # 查看当前已保留的缓存显存 (PyTorch 预分配但未使用的部分)
        reserved = torch.cuda.memory_reserved(device=None)
        logger.info(f"[===GPU MEMORY===] 已保留显存: {reserved / 1024**2:.2f} MB <=========")

    @override
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()  # 释放显存
        self.__get_gpu_memory("训练开始时")
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        # 数据加载器 和 训练步骤数
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        # 包装模型时，不要使用“accelerator.prepare”，这适用于未处理的情况，如FSDP-XLA、SageMaker MP/DP、DataParallel、IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        # 断点续训
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.state.init_training_references(self, train_dataloader, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        # tr_loss 是一个张量，用于避免TPUs通过 .item() 同步
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        # _total_loss_scalar 是每次调用 tr_loss 的 .item() 时都会更新，并存储所有损失的总和
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()  # 清空梯度
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        ## 这是MAML的外层循环
        ## 这是主要的修改之处
        for epoch in range(epochs_trained, num_train_epochs):
            """
            # 遍历每个用户（任务）
            # 每个任务的数据量: N-shot, K-query
            """
            logger.info_rank0(f"{'='*10}> 当前在第 {epoch} 个Epoch.")
            task_losses = []  # 用于存放每个任务的loss(标量), 最后用于更新Model
            self.__get_gpu_memory(f"第 {epoch} 个Epoch, 复制前")
            ############### 更新的代码 ###########
            ## 这里保存一份当前模型的参数，用于给后面 Task-Specific Model 初始化
            # 保存当前模型参数（元参数）作为备份
            meta_params = copy.deepcopy(model.state_dict())
            # 复制一份 self.optimizer 作为备份
            meta_optimizer = copy.deepcopy(self.optimizer.state_dict())
            # 复制一个 meta_accelerator 和 init_accelerator
            meta_accelerator = copy.deepcopy(self.accelerator)
            init_accelerator = copy.deepcopy(self.accelerator)
            ############### 更新的代码 ###########
            self.__get_gpu_memory(f"第 {epoch} 个Epoch, 复制后")

            self.accelerator.free_memory()  # 释放显存
            torch.cuda.empty_cache()  # 释放显存

            self.__get_gpu_memory(f"第 {epoch} 个Epoch, 释放显存后")

            for maml_task in range(self.maml_num_tasks):
                logger.info_rank0(f"{'='*10}> 当前在第 {epoch} 个Epoch的第 {maml_task} 个任务.")

                # --- 内循环：在支持集上做几步梯度更新 ---
                # 克隆一份模型供内循环使用，防止直接改动 model 参数
                # 在 MAML 内循环中，使用 model 和 self.optimizer 来训练 task-specific model

                self.__get_gpu_memory(f"在第 {epoch} 个Epoch的第 {maml_task} 个任务. Model 初始化前")
                ############### 更新的代码 ###########
                # Task-Specific Model 初始化
                model.load_state_dict(meta_params)
                self.optimizer.load_state_dict(meta_optimizer)
                # 重置优化器状态
                self.optimizer.state = collections.defaultdict(dict)
                # 重置加速器状态
                self.accelerator = init_accelerator
                ############### 更新的代码 ###########
                self.__get_gpu_memory(f"在第 {epoch} 个Epoch的第 {maml_task} 个任务. Model 初始化后")

                ## 数据加载器
                self.support_dataset, self.query_dataset = self.process_MAML_dataset(
                    self.maml_training_dataset_list[maml_task]
                )
                self.train_dataset = self.support_dataset  # 第maml_task个任务的 支持集
                train_dataloader = self.get_train_dataloader()  # 这个函数会读取 self.train_dataset
                if self.is_fsdp_xla_v2_enabled:
                    train_dataloader = tpu_spmd_dataloader(train_dataloader)
                epoch_dataloader = train_dataloader

                if hasattr(epoch_dataloader, "set_epoch"):
                    epoch_dataloader.set_epoch(epoch)

                #################### 当前task的支持集训练 内循环 START #############################
                step = -1
                update_step = -1
                for i in range(self.maml_inner_epochs):
                    # Reset the past mems state at the beginning of each epoch if necessary.
                    # 重置过去mems状态，在每次epoch开始时
                    if args.past_index >= 0:
                        self._past = None

                    steps_in_epoch = (
                        len(epoch_dataloader)
                        if len_dataloader is not None
                        else args.max_steps * args.gradient_accumulation_steps
                    )
                    self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                    if (
                        epoch == epochs_trained
                        and resume_from_checkpoint is not None
                        and steps_trained_in_current_epoch == 0
                    ):
                        self._load_rng_state(resume_from_checkpoint)

                    rng_to_sync = False
                    steps_skipped = 0
                    if steps_trained_in_current_epoch > 0:
                        # 创建`torch.utils.data.DataLoader`
                        # 将有效地跳过前几个num_batches。如果原始数据加载器是“StatefulDataLoader”，则不应使用。
                        epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                        steps_skipped = steps_trained_in_current_epoch
                        steps_trained_in_current_epoch = 0
                        rng_to_sync = True

                    # step = -1
                    epoch_iterator = iter(epoch_dataloader)
                    # We chunkify the epoch iterator into gradient accumulation steps `n` batches
                    # 我们将epoch迭代器分块为梯度累积步骤`n`批
                    remainder = num_examples % args.gradient_accumulation_steps
                    if remainder == 0:
                        remainder = args.gradient_accumulation_steps
                    # update_step = -1
                    # gradient_accumulation_steps 用于控制在更新模型参数之前累积多少个小批量（mini-batch）的梯度。
                    total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
                    if args.gradient_accumulation_steps == 1:
                        total_updates -= 1
                    for _ in range(total_updates):
                        update_step += 1
                        logger.info_rank0(
                            f"{'='*10}> 当前在第 {epoch} 个Epoch的第 {maml_task} 个任务的第 {update_step} update_step."
                        )
                        num_batches = (
                            args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                        )
                        # 获取当前批次的样本和批次中的项目数量
                        batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                        for i, inputs in enumerate(batch_samples):
                            step += 1
                            logger.info_rank0(
                                f"{'='*10}> 当前在第 {epoch} 个Epoch的第 {maml_task} 个任务的第 {step} Step."
                            )
                            do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                                step + 1
                            ) == steps_in_epoch
                            # Since we perform prefetching, we need to manually set sync_gradients
                            self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                            # 跟踪输入令牌的数量. 评估模型的性能和资源使用情况时可以使用
                            if self.args.include_num_input_tokens_seen:
                                main_input_name = getattr(self.model, "main_input_name", "input_ids")
                                if main_input_name not in inputs:
                                    logger.warning(
                                        "Tried to track the number of tokens seen, however the current model is "
                                        "not configured properly to know what item is the input. To fix this, add "
                                        "a `main_input_name` attribute to the model class you are using."
                                    )
                                else:
                                    # 提取输入令牌的数量
                                    input_tokens = inputs[main_input_name].numel()  # numel()函数返回张量中元素的数量。
                                    input_tokens = torch.tensor(
                                        input_tokens, device=self.args.device, dtype=torch.int64
                                    )  # 转换为PyTorch张量
                                    self.state.num_input_tokens_seen += (
                                        self.accelerator.gather(input_tokens).sum().cpu().item()
                                    )  # 累加输入令牌的数量
                            if rng_to_sync:
                                self._load_rng_state(resume_from_checkpoint)
                                rng_to_sync = False

                            # Skip past any already trained steps if resuming training
                            # 跳过任何已经训练过的步骤，如果正在恢复训练
                            if steps_trained_in_current_epoch > 0:
                                steps_trained_in_current_epoch -= 1
                                if steps_trained_progress_bar is not None:
                                    steps_trained_progress_bar.update(1)
                                if steps_trained_in_current_epoch == 0:
                                    self._load_rng_state(resume_from_checkpoint)
                                continue
                            elif steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.close()
                                steps_trained_progress_bar = None

                            if step % args.gradient_accumulation_steps == 0:
                                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                            # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                            # 我们明确地不想依赖 `accelerator.accumulate` 来进行生成训练
                            context = (
                                functools.partial(self.accelerator.no_sync, model=model)
                                if i != len(batch_samples) - 1
                                and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                                else contextlib.nullcontext
                            )
                            with context():
                                """
                                training_step() 包含模型的正向传播 和 反向更新, 但没有更新参数
                                这一步应该是用 support_x 和 support_y 来训练该Task的模型
                                """
                                tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                            if (
                                args.logging_nan_inf_filter
                                and not is_torch_xla_available()
                                and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                            ):
                                # 如果损失为nan或inf，只需将之前记录的损失的平均值相加
                                # if loss is nan or inf simply add the average of previous logged losses
                                tr_loss = tr_loss + tr_loss / (
                                    1 + self.state.global_step - self._globalstep_last_logged
                                )
                            else:
                                if tr_loss.device != tr_loss_step.device:
                                    raise ValueError(
                                        f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                    )
                                tr_loss = tr_loss + tr_loss_step

                            self.current_flos += float(self.floating_point_ops(inputs))

                            if do_sync_step:
                                # Since we perform prefetching, we need to manually set sync_gradients to True
                                # 由于我们执行预取，因此需要手动将sync_gradients设置为True
                                self.accelerator.gradient_state._set_sync_gradients(True)

                                # Gradient clipping
                                # 梯度裁剪
                                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                    if is_sagemaker_mp_enabled() and args.fp16:
                                        _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                    elif self.use_apex:
                                        # Revert to normal clipping otherwise, handling Apex or full precision
                                        # 否则，处理Apex或全精度，恢复到正常裁剪
                                        _grad_norm = nn.utils.clip_grad_norm_(
                                            amp.master_params(self.optimizer),
                                            args.max_grad_norm,
                                        )
                                    else:
                                        _grad_norm = self.accelerator.clip_grad_norm_(
                                            model.parameters(),
                                            args.max_grad_norm,
                                        )

                                    if (
                                        is_accelerate_available()
                                        and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                    ):
                                        grad_norm = model.get_global_grad_norm()
                                        # In some cases the grad norm may not return a float
                                        # 在这种情况下，梯度范数可能不会返回一个浮点数
                                        if hasattr(grad_norm, "item"):
                                            grad_norm = grad_norm.item()
                                    else:
                                        grad_norm = _grad_norm

                                # 回调函数
                                self.control = self.callback_handler.on_pre_optimizer_step(
                                    args, self.state, self.control
                                )

                                # 正常应该调用optimizer.zero_grad()，防止梯度累加干扰当前批次的计算
                                # 但梯度累加也可用于模拟大batch训练
                                self.optimizer.step()  # 更新 task-specific 模型参数

                                self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                                if not self.accelerator.optimizer_step_was_skipped:
                                    # Delay optimizer scheduling until metrics are generated
                                    # 延迟优化器调度，直到生成指标
                                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                        self.lr_scheduler.step()

                                model.zero_grad()  # 清空梯度
                                self.state.global_step += 1
                                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                                self._maybe_log_save_evaluate(
                                    tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                                )
                            else:
                                self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                            # PyTorch/XLA relies on the data loader to insert the mark_step for
                            # each step. Since we are breaking the loop early, we need to manually
                            # insert the mark_step here.
                            if self.control.should_epoch_stop or self.control.should_training_stop:
                                if is_torch_xla_available():
                                    xm.mark_step()
                                break
                        # We also need to break out of the nested loop
                        # 我们也需要从嵌套循环中跳出
                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            if is_torch_xla_available():
                                xm.mark_step()
                            break
                    if step < 0:
                        logger.warning(
                            "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                            f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                            f" num_steps ({max_steps}) higher than the number of available samples."
                        )
                        # 输出中文
                        logger.warning(
                            f"似乎你的epoch_iterator中没有单个样本，在步骤{self.state.global_step}处停止训练！如果你使用的是IterableDataset，并且将num_steps ({max_steps})设置得高于可用样本的数量，这是预期的。"
                        )
                        self.control.should_training_stop = True

                    logger.info_rank0(f"{'='*10}> 完成第 {epoch} 个Epoch的第 {maml_task} 个任务的support训练")

                    self.__get_gpu_memory(f"完成第 {epoch} 个Epoch的第 {maml_task} 个任务的support训练")

                    self.accelerator.free_memory()  # 释放显存
                    torch.cuda.empty_cache()  # 释放显存
                    self.__get_gpu_memory(f"完成第 {epoch} 个Epoch的第 {maml_task} 个任务的support训练, 并释放显存")

                #################### 当前task的支持集训练 内循环 END #############################

                ###################### 计算 Task-Specific Model 的 查询集(query)损失(带梯度) START ####################
                logger.info(f"\n***** Running Query *****")
                query_loss_list = []
                query_loss = torch.tensor(0.0).to(args.device)
                ## 用当前task的 查询集 计算 Model_i 的loss, 并保存（不在这里做反向传播）
                query_dataset = self.query_dataset  # 查询集
                query_dataloader = self.get_test_dataloader(query_dataset)  # 获取查询集的 dataloader
                if self.is_fsdp_xla_v2_enabled:
                    query_dataloader = tpu_spmd_dataloader(query_dataloader)
                epoch_dataloader = query_dataloader

                if hasattr(epoch_dataloader, "set_epoch"):
                    epoch_dataloader.set_epoch(epoch)

                # Reset the past mems state at the beginning of each epoch if necessary.
                # 重置过去mems状态，在每次epoch开始时
                # if args.past_index >= 0:
                #     self._past = None

                steps_in_epoch = (
                    len(epoch_dataloader)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
                # self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                if (
                    epoch == epochs_trained
                    and resume_from_checkpoint is not None
                    and steps_trained_in_current_epoch == 0
                ):
                    self._load_rng_state(resume_from_checkpoint)

                rng_to_sync = False
                steps_skipped = 0
                if steps_trained_in_current_epoch > 0:
                    # 创建`torch.utils.data.DataLoader`
                    # 将有效地跳过前几个num_batches。如果原始数据加载器是“StatefulDataLoader”，则不应使用。
                    epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                    steps_skipped = steps_trained_in_current_epoch
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

                step = -1
                epoch_iterator = iter(epoch_dataloader)
                # We chunkify the epoch iterator into gradient accumulation steps `n` batches
                # 我们将epoch迭代器分块为梯度累积步骤`n`批
                remainder = num_examples % args.gradient_accumulation_steps
                if remainder == 0:
                    remainder = args.gradient_accumulation_steps
                update_step = -1
                # gradient_accumulation_steps 用于控制在更新模型参数之前累积多少个小批量（mini-batch）的梯度。
                total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
                if args.gradient_accumulation_steps == 1:
                    total_updates -= 1
                for _ in range(total_updates):
                    update_step += 1
                    num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                    # 获取当前批次的样本和批次中的项目数量
                    batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                    for i, inputs in enumerate(batch_samples):
                        step += 1
                        do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                            step + 1
                        ) == steps_in_epoch
                        # Since we perform prefetching, we need to manually set sync_gradients
                        self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        # 我们明确地不想依赖 `accelerator.accumulate` 来进行生成训练
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples) - 1
                            and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        self.__get_gpu_memory(f"计算第 {step+1} query step 前")
                        with context():
                            """
                            training_step() 包含模型的正向传播 和 反向更新, 但没有更新参数
                            这一步应该是用 support_x 和 support_y 来训练该Task的模型
                            """
                            # query_loss_step = self.training_step(model, inputs, num_items_in_batch)
                            query_loss_step = self.training_step_only_forward(model, inputs, num_items_in_batch)

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(query_loss_step) or torch.isinf(query_loss_step))
                        ):
                            # 如果损失为nan或inf，只需将之前记录的损失的平均值相加
                            # if loss is nan or inf simply add the average of previous logged losses
                            # tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                            pass
                        else:
                            if query_loss.device != query_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {query_loss_step.device}"
                                )
                            # FIXME: query_loss_step 占用显存过大.
                            query_loss_list.append(query_loss_step)  # 这种方式占用显存太大
                        model.zero_grad()  # 清空梯度

                        # PyTorch/XLA relies on the data loader to insert the mark_step for
                        # each step. Since we are breaking the loop early, we need to manually
                        # insert the mark_step here.
                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            if is_torch_xla_available():
                                xm.mark_step()
                            break
                    # We also need to break out of the nested loop
                    # 我们也需要从嵌套循环中跳出
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                if step < 0:
                    logger.warning(
                        "======= query ==========="
                        "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    # 输出中文
                    logger.warning(
                        f"======= query ===========\n似乎你的epoch_iterator中没有单个样本，在步骤{self.state.global_step}处停止训练！如果你使用的是IterableDataset，并且将num_steps ({max_steps})设置得高于可用样本的数量，这是预期的。"
                    )
                    self.control.should_training_stop = True

                self.__get_gpu_memory(f"完成第 {epoch} 个Epoch的第 {maml_task} 个任务的query loss 计算")

                ###################### 计算 Task-Specific Model 的 查询集(query)损失(带梯度) END ####################

                query_loss = torch.stack(query_loss_list).mean()
                # 直接在这对 meta_accelerator 进行反向传播,累计梯度
                query_loss = query_loss / self.maml_num_tasks  # 损失归一化
                del query_loss_list  # 释放显存

                ###### kwargs START ########
                kwargs = {}

                # For LOMO optimizers you need to explicitly use the learnign rate
                if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                    kwargs["learning_rate"] = self._get_learning_rate()

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                meta_accelerator.backward(query_loss, **kwargs)
                del query_loss

                self.accelerator.free_memory()  # 释放显存
                torch.cuda.empty_cache()  # 释放显存
                self.__get_gpu_memory(
                    f"完成第 {epoch} 个Epoch的第 {maml_task} 个任务对 meta_accelerator 的梯度累加, 并释放显存"
                )

                ######### 计算 Task-Specific Model 的 损失(带梯度) ##########

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)
                logger.info(f">>>>>>> 完成第 {epoch} 个Epoch的第 {maml_task} 个任务的query推理 <<<<<<<<<")

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_xla_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                        # 输出中文
                        logger.warning(
                            "你启用了PyTorch/XLA调试指标，但你没有配置TPU。如果这是意外的，请检查你的训练配置。"
                        )
                if self.control.should_training_stop:
                    break
            ################################
            # 完成了1个epoch内所有Task的训练 #
            ################################
            logger.info_rank0(f"{'='*10}> 完成第 {epoch} 个Epoch内所有任务的训练, {self.maml_num_tasks=}.")

            self.__get_gpu_memory(f"完成第 {epoch} 个Epoch内所有任务的训练时， 还未还原Model和Optimizer")
            ############## 还原回 Epoch 开始时的 Model 和 Optimizer ##############
            model.load_state_dict(meta_params)
            self.optimizer.load_state_dict(meta_optimizer)
            self.accelerator = meta_accelerator
            self.__get_gpu_memory(f"完成第 {epoch} 个Epoch内所有任务的训练时， 还原回Epoch开始时的Model和Optimizer")

            self.optimizer.step()  # 更新 task-specific 模型参数
            # self.optimizer.zero_grad() # 清空梯度

            logger.info_rank0("完成了1个epoch内所有Task的训练")

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        logger.info("\n\n训练完成。不要忘记在huggingface.co/models上分享你的模型 =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            # 等待每个人到达这里，以确保进程0已经保存了模型。
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        # 添加剩余的 tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    # 根据 evaluation_loop 修改得来的 query_loop，用于在查询集上用 task-specific 模型计算loss
    def query_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
    ) -> float:
        """
        Description：
            根据 evaluation_loop 修改得来的 query_loop，用于在查询集上用 task-specific 模型计算loss

            Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
            Works both with or without labels.
        Args:
            prediction_loss_only (`bool`): Whether or not to return the loss only. 是否仅返回损失。
                - prediction_loss_only = True: return (loss, None, None)
                - prediction_loss_only = False: return (loss, logits, labels)
            ignore_keys: 模型输出中的一系列键（如果是字典），在收集预测时应该忽略这些键。
        Returns:
            result_loss (`float`): 损失
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()  # 将模型设置为评估模式
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # 将损失值、预测值、标签和输入值都收集到各自的容器中
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}  # 存放 losses 等参数，用于在计算评估指标时使用

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            ## Prediction step
            # prediction_loss_only == True 的话, losses, logits, labels
            # 会调用 CustomSeq2SeqTrainer.prediction_step
            # losses.requires_grad=False 没有梯度保留
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            logger.info(f">>>>>>>>>>>>>>>>>> {losses=} <<<<<<<<<<<<<<<<<<<<")
            logger.info(f">>>>>>>>>>>>>>>>>> {losses.requires_grad=} <<<<<<<<<<<<<<<<<<<<")
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers 更新容器
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)

            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            # 生成时不将标签传递给模型
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        result_loss = all_losses.tensors.mean()

        # Gather all remaining tensors and put them back on the CPU
        # 收集所有剩余的张量并将它们放回CPU
        all_losses = all_losses.get_arrays()  # 将损失值收集到 all_losses 容器中，并在最后计算平均损失
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        return result_loss

    def process_MAML_dataset(
        self,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]],
    ):
        """
        处理训练数据集，构造MAML的支持集和查询集。

        Args:
            train_dataset: 训练数据集，可以是 Dataset、IterableDataset 或 datasets.Dataset。
            num_shots: 支持集的样本数量。
            num_querys: 查询集的样本数量。

        Returns:
            support_dataset: 支持集数据集。 Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]]
            query_dataset: 查询集数据集。 Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]]
        """
        num_shots = self.num_shots
        num_querys = self.num_querys
        if train_dataset is None:
            logger.error("错误: train_dataset is None! Exit!")
            exit()

        # 确保样本数量不超过数据集大小
        total_samples = len(train_dataset)
        if num_shots + num_querys > total_samples:
            logger.error("错误: num_shots + num_querys > total_samples")
            exit()

        logger.info_rank0(f"{type(train_dataset)=}")

        shuffled_dataset = train_dataset.shuffle(seed=42)
        support_dataset = shuffled_dataset.select(range(0, num_shots))
        query_dataset = shuffled_dataset.select(range(num_shots, num_shots + num_querys))

        return support_dataset, query_dataset

    # @override
    def __prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            # 生成时不将标签传递给模型
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        result_labels = labels
        """
        loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs)
        """
        #########################################
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        # with torch.no_grad():
        if is_sagemaker_mp_enabled():
            raw_outputs = smp_forward_only(model, inputs)
            if has_labels or loss_without_labels:
                if isinstance(raw_outputs, dict):
                    loss_mb = raw_outputs["loss"]
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    loss_mb = raw_outputs[0]
                    logits_mb = raw_outputs[1:]

                # loss = loss_mb.reduce_mean().detach().cpu()
                loss = loss_mb.reduce_mean()  # 保留梯度
                logits = smp_nested_concat(logits_mb)
            else:
                loss = None
                if isinstance(raw_outputs, dict):
                    logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                else:
                    logits_mb = raw_outputs
                logits = smp_nested_concat(logits_mb)
        else:
            if has_labels or loss_without_labels:
                # 我的程序会走这一条分支
                self.__get_gpu_memory("In prediction_step")
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                # loss = loss.mean().detach()
                loss = loss.mean()  # 保留梯度

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        # return (loss, logits, labels)

        ########################################
        generated_tokens = logits
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        logger.info(f">>>>>>>>>>>>>>>>>>In prediction_step: {loss=}")
        logger.info(f">>>>>>>>>>>>>>>>>>In prediction_step: {loss.requires_grad=}")

        return loss, generated_tokens, result_labels

    @override
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            # "batch_size": self.args.eval_batch_size,
            "batch_size": 1,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def training_step_only_forward(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        抄的 Trainer.training_step
        对一批输入执行训练步骤。只有正向传播+反向传播
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
                字典在放入模型之前将被解包。大多数模型都期望在参数`labels`下找到targets。
                检查模型文档中所有可接受的参数。

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()  # 将模型设置为训练模式
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()  # 将优化器设置为训练模式

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            # 从 smp_options 变量中获取特定于 sagemaker 的 mp 参数。
            print("执行到了 MAMLSeq2SeqTrainer.training_step 的 is_sagemaker_mp_enabled() 中的 if 语句")
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # ​上下文管理器​​，用于​​在计算损失时自动管理混合精度训练、梯度累积、分布式训练等环境配置​​。
        with self.compute_loss_context_manager():
            # 正向传播过程
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs  # 释放内存

        # 如果设置了torch_empty_cache_steps参数，并且当前全局步数是torch_empty_cache_steps的倍数
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            # 如果是torch_xpu设备
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            # 如果是torch_mlu设备
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            # 如果是torch_musa设备
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            # 如果是torch_npu设备
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            # 如果是torch_mps设备，并且版本大于等于2.0
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            # 否则，使用torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()  # 释放内存
        self.accelerator.free_memory()  # 释放显存

        # 多GPU训练
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # kwargs = {}

        # # For LOMO optimizers you need to explicitly use the learnign rate
        # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        #     kwargs["learning_rate"] = self._get_learning_rate()

        # # 多GPU训练
        # if self.args.n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.use_apex:
        #     # 混合精度训练, 用 自动精度 包裹起来
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     # Finally we need to normalize the loss for reporting
        #     if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
        #         loss = loss / self.args.gradient_accumulation_steps

        #     # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
        #     # https://github.com/huggingface/transformers/pull/35808
        #     if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         kwargs["scale_wrt_gas"] = False

        #     self.accelerator.backward(loss, **kwargs)  # 反向传播

        #     return loss.detach()
        return loss


if __name__ == "__main__":
    # Initialize our Trainer
    # 初始化我们的Trainer
    dataset_module: Dict[str, "Dataset"] = {"eval_dataset": "", "train_dataset": ""}
    tokenizer_module = {"tokenizer": "tokenizer", "processor": "processor"}
    metric_module = {}

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
