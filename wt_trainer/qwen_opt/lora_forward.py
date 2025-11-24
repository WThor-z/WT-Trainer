import torch


def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

    result = self.base_layer(x, *args, **kwargs)

    if self.is_quantized_4bit_need_clone:
        result = result.clone()

    # 设定切片大小，根据显存动态调整，通常 4096 或 2048 比较安全
    CHUNK_SIZE = 2048

    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]

        requires_conversion = not torch.is_autocast_enabled()
        expected_dtype = result.dtype
        target_dtype = lora_A.weight.dtype

        for i in range(0, x.shape[0], CHUNK_SIZE):
            # Slice Input
            x_chunk = x[i : i + CHUNK_SIZE]

            if requires_conversion:
                x_chunk = x_chunk.to(target_dtype)

            chunk_out = lora_B(lora_A(dropout(x_chunk))) * scaling

            if requires_conversion:
                chunk_out = chunk_out.to(expected_dtype)

            # 直接加到 result 的对应位置
            result[i : i + CHUNK_SIZE].add_(chunk_out)

            del chunk_out
            del x_chunk

    return result
