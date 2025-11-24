from datetime import datetime
import os
from zoneinfo import ZoneInfo
from typing import Iterable, Callable, Optional
import logging

from torch._C._profiler import ProfilerActivity
from torch.profiler import profile, ProfilerAction, supported_activities, tensorboard_trace_handler
from torch.autograd.profiler import record_function

logger = logging.getLogger(__name__)


class MemoryMonitor(profile):
    """继承自Profiler上下文管理器"""

    def __init__(
        self,
        *,
        activities: Iterable[ProfilerActivity] | None = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
            ProfilerActivity.XPU,
        ],
        schedule: Callable[[int], ProfilerAction] | None = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        module_name: str = "default_module",
        with_flops: bool = False,
        with_modules: bool = False,
        record_target: Optional[bool] = None,
    ):
        # 倘若返回一组受支持的性能分析器追踪活动
        activities_set = set(activities) if activities else supported_activities()

        # 强制要求指出活动
        assert len(activities_set) > 0, "No valid profiler activities found"

        self.module_name = module_name
        beijing_tz = ZoneInfo("Asia/Shanghai")
        beijing_now = datetime.now(beijing_tz)
        timestamp = beijing_now.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("tensorboard_logs", f"Function-Test/{module_name}_{timestamp}")
        on_trace_ready = tensorboard_trace_handler(log_dir)

        super().__init__(
            activities=activities,
            record_shapes=record_shapes,
            schedule=schedule,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            on_trace_ready=on_trace_ready,
        )

        self.record_target = record_target
        self.target_module_set: set = set()

        self.module_rec_fn: CustomRecorder = None

    def __enter__(self):

        super().__enter__()

        if self.module_rec_fn is None:
            self.module_rec_fn = CustomRecorder(self.module_name, mmoniter=self)
            self.module_rec_fn.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.module_rec_fn:
            self.module_rec_fn.__exit__(None, None, None)

        super().__exit__(exc_type, exc_val, exc_tb)

        if self.record_target != None:
            logger.info("Model profiling results:")
            # 获取所有事件的统计信息

            target_events = self._get_registered_event
            stats = self.key_averages()

            table_info = stats.table(sort_by="cuda_memory_usage", row_limit=-1)
            if not target_events:
                logger.info(table_info)
            else:
                lines = table_info.split("\n")

                header = lines[1]
                body_lines = lines[3:]

                import re

                pattern = r"\S+"

                lines = table_info.split("\n")
                separator = lines[0]  # 第一条分隔线
                header = lines[1]  # 表头
                separator2 = lines[2]  # 第二条分隔线
                body_lines = lines[3:-1]  # 去掉最后的分隔线

                # 过滤出包含目标事件的行
                filtered_body = [
                    line
                    for line in body_lines
                    if any(
                        target
                        in [
                            re.search(pattern, line).group() if re.search(pattern, line) else "None"
                        ]
                        for target in target_events
                    )
                ]

                if not filtered_body:
                    logger.info("No matching events found for the specified targets.")
                    logger.info("Displaying full table instead:")
                    logger.info(table_info)
                else:
                    # 重新拼接成表格格式
                    final_table_str = "\n".join(
                        [separator, header, separator2] + filtered_body + [separator]
                    )
                    logger.info(final_table_str)

            self.export_memory_timeline("gpu_memory_usage.html")

    def register_event(self, event_name: str):
        """
        Register a special event to be tracked.
        """
        if self.record_target == True:
            self.target_module_set.add(event_name)

    @property
    def _get_registered_event(self):
        """
        Get the registered event names.
        """
        return list(self.target_module_set)


class CustomRecorder(record_function):
    """
    继承自record_function，将其本地化，主要功能是每次记录时都会注册到MemoryMoniter的特殊标记集合中
    """

    def __init__(
        self,
        name: str,
        args: Optional[str] = None,
        mmoniter: Optional[MemoryMonitor] = None,
    ):
        super().__init__(name, args)
        if mmoniter:
            mmoniter.register_event(name)
