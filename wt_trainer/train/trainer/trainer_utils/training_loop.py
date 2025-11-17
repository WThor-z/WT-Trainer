"""Supervised Fine-Tuning (SFT) training loop implementation.

This module contains the core training loop for supervised fine-tuning of models.
It handles the complete training process including data loading, gradient accumulation,
optimizer steps, and checkpoint management.

The training loop follows these main steps:
1. Setup model and training parameters
2. Iterate through epochs
3. For each epoch, iterate through batches with gradient accumulation
4. Perform optimizer steps and learning rate scheduling
5. Handle logging, evaluation, and checkpointing
"""

import contextlib
import functools
import logging
import os
import shutil
import time
from typing import Any

from accelerate import DistributedType
import torch
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import speed_metrics
from transformers.trainer_utils import TrainOutput

from wt_trainer.args import TrainingArguments

from .training_loop_utils import preprocess
from .training_loop_utils import training_step

logger = logging.getLogger(__name__)


def sft_train(
    self,
    batch_size: int | None = None,
    args: TrainingArguments | None = None,
    **kwargs: Any,
) -> TrainOutput:
    """Execute the supervised fine-tuning training loop.

    Args:
        self: Trainer instance containing model and training components.
        batch_size: Training batch size. If None, will use args.per_device_train_batch_size.
        args: Training arguments. If None, will use self.args.
        **kwargs: Additional keyword arguments.

    Returns:
        TrainOutput: Contains the number of steps, training loss, and metrics.

    Raises:
        RuntimeError: If training encounters an unrecoverable error.
    """
    # Ensure model is on the correct device
    if hasattr(self.model, "to") and hasattr(args, "device"):
        self.model = self.model.to(args.device)
    elif hasattr(self.model, "to") and hasattr(args, "_setup_devices"):
        # If we can't access device attribute directly, try using _setup_devices
        self.model = self.model.to(args._setup_devices)

    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
        total_train_batch_size,
        train_dataloader,
    ) = preprocess(self, batch_size, args)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Num Epochs = {num_train_epochs:,}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_steps:,}")
    logger.info(
        f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}"
    )

    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0  # Current number of trained epochs
    steps_trained_in_current_epoch = 0  # Current number of trained steps
    steps_trained_progress_bar = None

    # Define loss
    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0, device=args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step

    # Reset model state if improper operations were performed previously
    self.model.zero_grad()

    grad_norm: float | None = None
    learning_rate = None

    # Setup callback for on_train_begin
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    # Enter epoch training
    for epoch in range(epochs_trained, num_train_epochs):
        # Set a dedicated dataloader for each epoch, which can satisfy dynamic data adjustment
        # and different training methods for different epochs
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

        # Callback at the beginning of epoch
        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

        epoch_iterator = iter(epoch_dataloader)
        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % args.gradient_accumulation_steps
        if remainder == 0:
            remainder = args.gradient_accumulation_steps

        total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
            remainder < args.gradient_accumulation_steps
        )

        # Enter step training for the epoch
        for update_step in range(total_updates):

            batch_num = (
                args.gradient_accumulation_steps
                if update_step != (total_updates - 1)
                else remainder
            )  # Number of samples required for each step

            batch_samples, num_items_in_batch = self.get_batch_samples(
                epoch_iterator, batch_num, args.device
            )

            # Store the number of batches for current gradient accumulation
            # This is used to correctly scale the loss when the last accumulation step has fewer batches
            self.current_gradient_accumulation_steps = len(batch_samples)

            # Enter training for each sample in the step
            for sample_index, sample in enumerate(batch_samples):

                # This is a flag that determines whether to update parameters
                do_optim_update = (
                    sample_index % args.gradient_accumulation_steps == 0
                    or (sample_index + 1) == steps_in_epoch
                )

                self.accelerator.gradient_state._set_sync_gradients(do_optim_update)

                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                # Callback at the beginning of step
                if sample_index % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                # self.accelerator.no_sync: This context manager's role is to temporarily disable gradient synchronization.
                # Backpropagation executed within its scope will be prevented from synchronizing gradients even if the underlying DDP wants to.
                # This is exactly what we want: gradients are only accumulated locally.
                context = (
                    functools.partial(self.accelerator.no_sync, model=self.model)
                    if sample_index != len(batch_samples) - 1
                    and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                    else contextlib.nullcontext  # Otherwise provide an empty context manager.
                )

                with context():
                    tr_loss_step = training_step(self, self.model, sample)

                # Calculate loss
                if args.logging_nan_inf_filter and (
                    torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss = tr_loss + tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss = tr_loss + tr_loss_step

                # Estimate the computation amount (FLOPs) of the current training step,
                # which is useful for evaluating model efficiency and comparing computational costs of different models or training strategies
                self.current_flos += float(self.floating_point_ops(sample))
                if do_optim_update:
                    # Since we perform prefetching, we need to manually set sync_gradients to True
                    self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping, mainly used to prevent gradient explosion problems during training.
                    # When the norm of gradients becomes too large, directly updating parameters with these large gradients
                    # may cause the training process to become unstable or even diverge.
                    # By limiting the norm of gradients to a reasonable range (i.e., args.max_grad_norm),
                    # gradient clipping can stabilize the training process.
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        grad_norm_context = contextlib.nullcontext
                        with grad_norm_context():
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                args.max_grad_norm,
                            )

                    self.control = self.callback_handler.on_pre_optimizer_step(
                        args, self.state, self.control
                    )

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(
                        args, self.state, self.control
                    )

                    # get learning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            # Update the learning rate (lr) of parameters in the optimizer according to the predetermined learning rate scheduling strategy
                            self.lr_scheduler.step()

                    self.model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + sample_index / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        tr_loss=tr_loss,
                        grad_norm=grad_norm,
                        model=self.model,
                        trial=None,
                        epoch=epoch,
                        ignore_keys_for_eval=None,
                        start_time=start_time,
                        learning_rate=learning_rate,
                    )

                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(
            tr_loss=tr_loss,
            grad_norm=grad_norm,
            model=self.model,
            trial=None,
            epoch=epoch,
            ignore_keys_for_eval=None,
            start_time=start_time,
            learning_rate=learning_rate,
        )

        if self.control.should_training_stop:
            break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    logger.info("\n\nTraining completed. See you next time!\n\n")

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    train_loss = self._total_loss_scalar / effective_global_step

    metrics = speed_metrics(
        "train",
        start_time,
        num_samples=num_train_samples,
        num_steps=self.state.max_steps,
    )
    self.store_flos()
    metrics["total_flos"] = self.state.total_flos
    metrics["train_loss"] = train_loss

    self.is_in_train = False

    self._memory_tracker.stop_and_update_metrics(metrics)

    self.log(metrics)

    run_dir = self.args.output_dir
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    if (
        self.args.should_save
        and self.state.best_model_checkpoint is not None
        and self.args.save_total_limit == 1
    ):
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(
                    f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                )
                shutil.rmtree(checkpoint, ignore_errors=True)

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    return TrainOutput(self.state.global_step, train_loss, metrics)