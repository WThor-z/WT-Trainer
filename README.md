# WT-Trainer

WT-Trainer is a deep learning model fine-tuning framework based on PyTorch, focusing on fine-tuning tasks for large
language models. It supports multiple fine-tuning methods including LoRA and full-parameter fine-tuning.

[中文文档](doc/README_zh.md) | [English Documentation](README.md)

## Features

- **Multi-Model Support**: Supports various large language model architectures (e.g., Qwen, Llama)
- **Multiple Fine-tuning Methods**: Supports LoRA, QLoRA, full-parameter fine-tuning and more
- **Data Processing**: Built-in support for multiple dataset formats and data preprocessing tools
- **Distributed Training**: Supports multi-GPU distributed training
- **Model Quantization**: Supports 4-bit and 8-bit model quantization to reduce memory usage

## Installation

```bash
# Clone the project
git clone https://github.com/your-username/WT-Trainer.git
cd WT-Trainer

# Create virtual environment
conda create -n wt_trainer python=3.9
conda activate wt_trainer

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Training Parameters

Modify parameters in [wt_trainer/train/llama3_lora_sft.yaml](wt_trainer/train/llama3_lora_sft.yaml):

```yaml
# Model Configuration
model_name_or_path: /path/to/your/model  # Model path
trust_remote_code: true

# Fine-tuning Method Configuration
stage: sft  # Training stage
do_train: true
finetuning_type: lora  # Fine-tuning type
lora_rank: 8  # LoRA rank
lora_target: all  # LoRA target layers

# Dataset Configuration
dataset: in_100  # Dataset name
template: qwen3  # Template type
cutoff_len: 2048  # Maximum sequence length
```

### 2. Start Training

```bash
# Run training script
python train.py --config_path wt_trainer/train/llama3_lora_sft.yaml
```

## Project Structure

```
WT-Trainer/
├── data/                   # Dataset directory
├── plugin/                 # Plugin directory
├── test/                   # Test directory
├── wt_trainer/             # Core code directory
│   ├── args/               # Parameter definitions
│   ├── train/              # Training related code
│   └── utils/              # Utility functions
├── doc/                    # Documentation directory
├── TASK_LIST.yaml          # Task list
├── README.md              # Project documentation (English)
├── requirements.txt       # Dependency list
├── train.py               # Main training script
└── install.sh/install.bat # Installation scripts
```

## Configuration Guide

### Model Arguments (model_args.py)

- `model_name_or_path`: Pretrained model path
- `cache_dir`: Cache directory
- `use_fast_tokenizer`: Whether to use fast tokenizer

### Fine-tuning Arguments (finetune_args.py)

- `finetuning_type`: Fine-tuning type (lora, full, freeze)
- `num_layer_trainable`: Number of trainable layers
- `lora_rank`: LoRA rank

### Training Arguments (training_args.py)

- `output_dir`: Output directory
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Training batch size
- `gradient_accumulation_steps`: Gradient accumulation steps

## Dataset Format

Multiple dataset formats are supported, including:

- JSON format: Contains instruction, input, output fields
- Custom format: Converted via data processors