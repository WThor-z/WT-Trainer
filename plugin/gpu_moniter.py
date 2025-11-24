# 通常 nvidia-ml-py 会将 pynvml 作为顶层模块提供
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    print("警告: 未安装 nvidia-ml-py，无法获取系统级GPU状态。请运行: pip install nvidia-ml-py")
    NVML_AVAILABLE = False

import time

import torch
from tqdm import tqdm


def tqdm_gpu_monitor_system_level(device_ids=[0], interval=1.0):
    """监控整个GPU设备的内存使用情况 (系统级)"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    if not NVML_AVAILABLE:
        return

    pynvml.nvmlInit()

    bars = {}
    handles = {}

    for idx, dev_id in enumerate(device_ids):
        # 使用 NVML 获取总内存，这更准确
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        nvml_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory_bytes = nvml_memory_info.total
        total_mem_gb = total_memory_bytes / (1024**3)

        bar = tqdm(
            desc=f"GPU {dev_id}",
            total=total_mem_gb,
            unit="GB",
            position=idx,
            leave=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total_fmt} [{postfix}]",
        )
        bars[dev_id] = bar
        handles[dev_id] = handle

    try:
        while True:
            for dev_id in device_ids:
                bar = bars[dev_id]
                handle = handles[dev_id]

                # 获取系统级内存信息 (所有进程)
                nvml_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_used_gb = nvml_memory_info.used / (1024**3)

                # 获取当前进程的 PyTorch 内存信息 (可能为0，如果当前进程没用GPU)
                pytorch_allocated_gb = torch.cuda.memory_allocated(dev_id) / (1024**3)
                pytorch_reserved_gb = torch.cuda.memory_reserved(dev_id) / (1024**3)

                # 计算利用率
                util = min(100, (total_used_gb / bar.total) * 100) if bar.total > 0 else 0
                color = "red" if util > 90 else "yellow" if util > 70 else "green"

                bar.set_postfix_str(
                    f"Sys Total: {total_used_gb:.2f}GB | PyTorch Alloc: {pytorch_allocated_gb:.2f}GB | Reserved: {pytorch_reserved_gb:.2f}GB",
                    refresh=False,
                )
                bar.n = total_used_gb
                bar.colour = color
                bar.refresh()  # 强制刷新

            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        for bar in bars.values():
            bar.close()
        if NVML_AVAILABLE:
            pynvml.nvmlShutdown()
        print("\n监控已安全退出")


# 使用示例
if __name__ == "__main__":
    tqdm_gpu_monitor_system_level(device_ids=[0], interval=0.5)
