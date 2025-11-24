from typing import Optional

import torch


class TempScope:

    def __init__(self, name: Optional[str] = None, empty_cache: bool = True):
        self.name = name or f"TempScope-{id(self):x}"
        self.empty_cache = empty_cache
        self.__urgent_tensor = None

    def __enter__(self) -> "TempScope":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__dict__.clear()

        if self.empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def add(self, **kwargs):
        name, tensor = next(iter(kwargs.items()))
        self.name = tensor

    def uadd(self, **kwargs):
        name, tensor = next(iter(kwargs.items()))
        self.name = tensor
        if self.__urgent_tensor:
            self.clear()
        self.__urgent_tensor = tensor

    def clear(self):

        del self.__urgent_tensor
