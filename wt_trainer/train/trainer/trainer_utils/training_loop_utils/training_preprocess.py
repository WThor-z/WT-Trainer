from transformers.modeling_utils import unwrap_model

from wt_trainer.args import TrainingArguments


def training_preprocess(
    self,
    batch_size: int | None = None,
    args: TrainingArguments | None = None,
) -> tuple:
    """
    Preprocesses the model and data for training.
    
    This function prepares the model and data loaders for training by:
    1. Freeing memory
    2. Setting up training parameters
    3. Creating optimizer and scheduler
    4. Wrapping the model
    
    Args:
        self: The trainer instance.
        batch_size: The batch size for training. If None, uses the existing batch size.
        args: Training arguments. If None, uses the existing arguments.
        
    Returns:
        A tuple containing:
        - num_train_epochs: Number of training epochs
        - num_update_steps_per_epoch: Number of update steps per epoch
        - num_examples: Number of examples
        - num_train_samples: Number of training samples
        - epoch_based: Whether training is epoch-based
        - len_dataloader: Length of the dataloader
        - max_steps: Maximum training steps
        - adjusted_batch_size: Batch size adjusted for gradient accumulation
        - train_dataloader: The training dataloader
    """
    self.accelerator.free_memory()
    self._train_batch_size = batch_size if batch_size else self._train_batch_size

    train_dataloader = self.get_train_dataloader()

    # Set initial training values without considering TP and DP
    (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
    ) = self.set_initial_training_values(
        args, train_dataloader, self._train_batch_size * args.gradient_accumulation_steps
    )

    # Reset learning rate scheduler if it was previously created
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False

    # Create optimizer and scheduler since we're not using distributed training
    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    self.state.train_batch_size = self._train_batch_size

    # Compute absolute values for logging, eval, and save if given as ratio
    self.state.compute_steps(args, max_steps)

    if args.gradient_checkpointing:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
        )

    # Wrap model to handle special cases like FSDP-XLA, SageMaker MP/DP, etc.
    # In case of auto_find_batch_size=True, remove FSDP wrapping from sub-models
    model = self._wrap_model(self.model_wrapped)

    use_accelerator_prepare = model is self.model
    if use_accelerator_prepare and self.is_fsdp_enabled:
        self.model = unwrap_model(self.model, recursive=True)

    # Prepare using `accelerator` prepare
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            # Prepare model and optimizer with accelerator
            # This handles device placement, distributed strategies, mixed precision, etc.
            model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            # Handle cases with "DummyScheduler" such as when specified in DeepSpeed config
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )

    # Update model_wrapped if the model was wrapped
    if model is not self.model:
        self.model_wrapped = model

    # Update the references of the callback handler
    for attr in ("model", "optimizer", "lr_scheduler"):
        setattr(self.callback_handler, attr, getattr(self, attr))
    self.callback_handler.train_dataloader = train_dataloader

    # Update train state
    self.state.max_steps = max_steps
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    return (
        num_train_epochs,
        num_update_steps_per_epoch,
        num_examples,
        num_train_samples,
        epoch_based,
        len_dataloader,
        max_steps,
        self._train_batch_size * args.gradient_accumulation_steps,
        train_dataloader,
    )