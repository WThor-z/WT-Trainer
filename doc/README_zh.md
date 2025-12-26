# WT-Trainer (中文版)

WT-Trainer是一个基于PyTorch的深度学习模型微调训练框架，专注于大语言模型的微调任务，支持LoRA、全量参数微调等多种微调方法。

## 功能特性

- **多模型支持**：支持多种大语言模型架构（如Qwen、Llama等）
- **多种微调方法**：支持LoRA、QLoRA、全量参数微调等方法
- **数据处理**：内置多种数据集格式支持和数据预处理工具
- **分布式训练**：支持多GPU分布式训练
- **模型量化**：支持4位、8位模型量化以减少显存占用

## 安装

```bash
# 克隆项目
git clone https://github.com/your-username/WT-Trainer.git
cd WT-Trainer

# 创建虚拟环境
conda create -n wt_trainer python=3.9
conda activate wt_trainer

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 配置训练参数

修改 [wt_trainer/train/llama3_lora_sft.yaml](wt_trainer/train/llama3_lora_sft.yaml) 文件中的参数：

```yaml
# 模型配置
model_name_or_path: /path/to/your/model  # 模型路径
trust_remote_code: true

# 微调方法配置
stage: sft  # 训练阶段
do_train: true
finetuning_type: lora  # 微调类型
lora_rank: 8  # LoRA等级
lora_target: all  # LoRA目标层

# 数据集配置
dataset: in_100  # 数据集名称
template: qwen3  # 模板类型
cutoff_len: 2048  # 序列最大长度
```

### 2. 启动训练

```bash
# 运行训练脚本
python train.py --config_path wt_trainer/train/llama3_lora_sft.yaml
```

## 项目结构

```
WT-Trainer/
├── data/                   # 数据集目录
├── plugin/                 # 插件目录
├── test/                   # 测试目录
├── wt_trainer/             # 核心代码目录
│   ├── args/               # 参数定义
│   ├── train/              # 训练相关代码
│   └── utils/              # 工具函数
├── doc/                    # 文档目录
├── TASK_LIST.yaml          # 任务清单
├── README.md              # 项目说明（英文版）
├── requirements.txt       # 依赖列表
├── train.py               # 主训练脚本
└── install.sh/install.bat # 安装脚本
```

## 配置说明

### 模型参数 (model_args.py)

- `model_name_or_path`: 预训练模型路径
- `cache_dir`: 缓存目录
- `use_fast_tokenizer`: 是否使用快速分词器

### 微调参数 (finetune_args.py)

- `finetuning_type`: 微调类型（lora, full, freeze）
- `num_layer_trainable`: 可训练层数量
- `lora_rank`: LoRA秩

### 训练参数 (training_args.py)

- `output_dir`: 输出目录
- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 训练批次大小
- `gradient_accumulation_steps`: 梯度累积步数

## 数据集格式

支持多种数据集格式，包括：

- JSON格式：包含instruction, input, output字段
- 自定义格式：通过数据处理器进行转换