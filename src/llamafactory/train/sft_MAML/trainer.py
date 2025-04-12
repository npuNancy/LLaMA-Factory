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
        基于 CustomSeq2SeqTrainer 修改的元学习Trainer
    Args:
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

    def __init__(self, maml_train_dataset_list, *args, **kwargs):
        """
        Args:
            maml_train_dataset_list: List[train_dataset]
        """
        super().__init__(*args, **kwargs)
        self.maml_train_dataset_list = maml_train_dataset_list  # MAML 训练任务列表
        self.maml_num_tasks = len(maml_train_dataset_list)  # MAML 训练任务的数量
        self.maml_inner_epochs = 5  # MAML 每个任务的训练轮次

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train and not self.is_model_parallel:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
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
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        # 这是MAML的外层循环
        # TODO: 这是主要的修改之处
        for epoch in range(epochs_trained, num_train_epochs):
            """
            # 遍历每个用户（任务）
            # 每个任务的数据量: N-shot, K-query
            """
            for maml_task in range(self.maml_num_tasks):

                # TODO 这里应该复制一个 model 和 optimizer, 用来完成这个Task的训练

                ## 数据加载器
                self.train_dataset = self.maml_train_dataset_list[maml_task]  # 第maml_task个任务的数据集
                train_dataloader = self.get_train_dataloader()  # 这个函数会读取 self.train_dataset.
                if self.is_fsdp_xla_v2_enabled:
                    train_dataloader = tpu_spmd_dataloader(train_dataloader)
                epoch_dataloader = train_dataloader

                if hasattr(epoch_dataloader, "set_epoch"):
                    epoch_dataloader.set_epoch(epoch)
                train_dataloader = self.get_train_dataloader()
                if self.is_fsdp_xla_v2_enabled:
                    train_dataloader = tpu_spmd_dataloader(train_dataloader)
                epoch_dataloader = train_dataloader

                if hasattr(epoch_dataloader, "set_epoch"):
                    epoch_dataloader.set_epoch(epoch)

                # Reset the past mems state at the beginning of each epoch if necessary.
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
                    epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                    steps_skipped = steps_trained_in_current_epoch
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

                step = -1
                epoch_iterator = iter(epoch_dataloader)
                # We chunkify the epoch iterator into gradient accumulation steps `n` batches
                remainder = num_examples % args.gradient_accumulation_steps
                if remainder == 0:
                    remainder = args.gradient_accumulation_steps
                update_step = -1
                total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
                if args.gradient_accumulation_steps == 1:
                    total_updates -= 1
                for _ in range(total_updates):
                    update_step += 1
                    num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                    batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                    for i, inputs in enumerate(batch_samples):
                        step += 1
                        do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                            step + 1
                        ) == steps_in_epoch
                        # Since we perform prefetching, we need to manually set sync_gradients
                        self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                        if self.args.include_num_input_tokens_seen:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            if main_input_name not in inputs:
                                logger.warning(
                                    "Tried to track the number of tokens seen, however the current model is "
                                    "not configured properly to know what item is the input. To fix this, add "
                                    "a `main_input_name` attribute to the model class you are using."
                                )
                            else:
                                input_tokens = inputs[main_input_name].numel()
                                input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                                self.state.num_input_tokens_seen += (
                                    self.accelerator.gather(input_tokens).sum().cpu().item()
                                )
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False

                        # Skip past any already trained steps if resuming training
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
                            tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
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
                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                            # 正常应该调用optimizer.zero_grad()，防止梯度累加干扰当前批次的计算
                            # 但梯度累加也可用于模拟大batch训练
                            self.optimizer.step()  # 更新模型参数

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

                """
                这里完成了当前task的训练
                用当前task的 查询集 计算原始模型的loss, 并保存（不在这里做反向传播）
                """

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_xla_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

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

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        对一批输入执行训练步骤。正常传播+反向传播, 但没有更新参数


        Subclass and override to inject custom behavior.
        子类和重写以注入自定义行为。

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

        # ​​上下文管理器​​，用于​​在计算损失时自动管理混合精度训练、梯度累积、分布式训练等环境配置​​。
        with self.compute_loss_context_manager():
            # 正向传播过程
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        # 多GPU训练
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            # 混合精度训练, 用 自动精度 包裹起来
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)  # 反向传播

            return loss.detach()


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
