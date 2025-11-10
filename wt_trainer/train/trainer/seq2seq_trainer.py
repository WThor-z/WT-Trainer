"""Sequence-to-sequence trainer with custom functionality."""

import logging
from types import MethodType
from typing import Any, TYPE_CHECKING

import torch
from torch import nn
from transformers import BaseImageProcessor
from transformers import FeatureExtractionMixin
from transformers import PreTrainedTokenizerBase
from transformers import Seq2SeqTrainer
from typing_extensions import override
from transformers import ProcessorMixin  # noqa: F401

from .trainer_utils import create_custom_optimizer
from .trainer_utils import create_custom_scheduler
from .trainer_utils import custom_compute_loss
from wt_trainer.utils.callbacks import SaveProcessorCallback
from wt_trainer.args import FinetuningArguments  # noqa: F401

if TYPE_CHECKING:
    from torch.utils.data import Dataset  # noqa: F401
    from transformers import PreTrainedTokenizer  # noqa: F401
    from transformers.trainer import PredictionOutput  # type: ignore # noqa: F401


logger = logging.getLogger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: FinetuningArguments,
        processing_class: (
            PreTrainedTokenizerBase
            | BaseImageProcessor
            | FeatureExtractionMixin
            | ProcessorMixin
            | None
        ),
        gen_kwargs: dict[str, Any] | None = None,
        use_custom_loss: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the custom sequence-to-sequence trainer.

        Args:
            finetuning_args: Fine-tuning arguments.
            processing_class: Processing class for tokenization or feature extraction.
            gen_kwargs: Generation keyword arguments.
            use_custom_loss: Whether to use custom loss function.
            **kwargs: Additional keyword arguments.
        """
        self.processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin
        ) | None = processing_class

        super().__init__(**kwargs)

        if processing_class is not None:
            # Avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processing_class is not None:
            self.add_callback(SaveProcessorCallback(processing_class))

        if finetuning_args.use_badam:
            from badam import BAdamCallback
            from badam import clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_old_version, self.accelerator
            )
            self.add_callback(BAdamCallback)

        self.use_custom_loss: bool = use_custom_loss

    @override
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create custom optimizer if needed.

        Returns:
            Optimizer instance.
        """
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create custom scheduler.

        Args:
            num_training_steps: Number of training steps.
            optimizer: Optimizer instance.

        Returns:
            Scheduler instance.
        """
        create_custom_scheduler(self.args, num_training_steps)
        return super().create_scheduler(num_training_steps, optimizer)  # type: ignore

    @override
    def _get_train_sampler(self, *args: Any, **kwargs: Any) -> torch.utils.data.Sampler | None:
        """Get train sampler, potentially disabling shuffling.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Train sampler instance or None.
        """
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], *args: Any, **kwargs: Any
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]] | None:
        """Compute loss using either custom or default implementation.

        Args:
            model: Model instance.
            inputs: Input data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Computed loss or None.
        """
        if self.use_custom_loss:
            return custom_compute_loss(self, model, inputs, *args, **kwargs)
        return super().compute_loss(model, inputs, *args, **kwargs)
