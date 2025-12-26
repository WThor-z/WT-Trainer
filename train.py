#!/usr/bin/env python
# coding=utf-8
"""
WT-Trainer 主训练脚本

该脚本用于启动WT-Trainer的训练流程。
"""

import argparse
import logging
from wt_trainer.train.training_control import run_train


def main():
    parser = argparse.ArgumentParser(description="WT-Trainer: A fine-tuning framework for large language models")
    parser.add_argument(
        "--config_path",
        type=str,
        default="wt_trainer/train/llama3_lora_sft.yaml",
        help="Path to the configuration file (default: wt_trainer/train/llama3_lora_sft.yaml)"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s | %(asctime)s] %(name)s  >>>>  %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # 运行训练
    run_train(config_file_path=args.config_path)


if __name__ == "__main__":
    main()