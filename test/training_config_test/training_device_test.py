import os

import torch
import accelerate
from transformers import TrainingArguments

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"ACCELERATE_USE_CPU: {os.environ.get('ACCELERATE_USE_CPU', 'Not Set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

# 尝试重置 Accelerate 状态
accelerate.state.AcceleratorState._reset_state(reset_partial_state=True)

# 创建 TrainingArguments
training_args = TrainingArguments(output_dir="./test_output")

# 访问 device 属性以触发 _setup_devices，这会缓存结果
device_prop = training_args.device
print(f"Final cached device: {device_prop}")
print(f"Final cached n_gpu: {training_args.n_gpu}")

# 检查 distributed_state 的最终状态
print(f"distributed_state.device: {training_args.distributed_state.device}")
print(f"distributed_state.distributed_type: {training_args.distributed_state.distributed_type}")
print(
    f"distributed_state.local_process_index: {training_args.distributed_state.local_process_index}"
)

# 尝试直接访问 PartialState 的一些属性
partial_state = training_args.distributed_state
print(f"PartialState.__dict__ keys: {list(partial_state.__dict__.keys())}")
try:
    # 尝试打印一些内部状态，具体键名可能因 accelerate 版本而异
    print(f"PartialState._cpu: {partial_state._cpu}")
    print(f"PartialState._mixed_precision: {partial_state._mixed_precision}")
    print(f"PartialState._backend: {partial_state._backend}")
except AttributeError as e:
    print(f"Could not access some PartialState attributes: {e}")
