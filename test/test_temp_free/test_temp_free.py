import sys

import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

sys.path.insert(0, "/home/inspur/zhawentao_workspace/WT_Trainer")
from wt_trainer.qwen_opt.rope import forward
from plugin.usage_moniter import MemoryMonitor

config = AutoConfig.from_pretrained("/home/inspur/zhawentao_workspace/Qwen3_8B/")

rotary_emb_orig = Qwen3RotaryEmbedding(config, device="cuda:0")
rotary_emb_optim = Qwen3RotaryEmbedding(config, device="cuda:0")

from functools import partial

rotary_emb_optim.forward = partial(forward, rotary_emb_optim)

batch_size, seq_len = 5, 1000
position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
position_ids = position_ids.to("cuda")

# 随便一个 hidden_states，只是为了拿 device 和 dtype
x = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda")

with MemoryMonitor(
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    module_name="Function_test_rope_orig",
) as mmoniter:
    with torch.no_grad():
        cos_orig, sin_orig = rotary_emb_orig(x, position_ids)

with MemoryMonitor(
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    module_name="Function_test_rope_optim",
) as mmoniter:
    with torch.no_grad():
        cos_yours, sin_yours = rotary_emb_optim(x, position_ids)

print("cos max diff:", (cos_orig - cos_yours).abs().max().item())
print("sin max diff:", (sin_orig - sin_yours).abs().max().item())
