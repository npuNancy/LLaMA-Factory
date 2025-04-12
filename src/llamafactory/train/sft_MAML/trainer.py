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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
