import torch
from transformers import dynamic_rope_update

from plugin.res_manager import TempScope


@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def forward(self, x, position_ids):
    inv_freq_expanded = (
        self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = (
        x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        with TempScope(name="rope") as tmp:
            tmp.uadd(
                freqs=(inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            )
            cos_half = tmp.freqs.cos().mul_(self.attention_scaling)
            sin_half = tmp.freqs.sin().mul_(self.attention_scaling)

        cos_half = cos_half.to(dtype=x.dtype)
        sin_half = sin_half.to(dtype=x.dtype)

    return torch.cat((cos_half, cos_half), dim=-1), torch.cat((sin_half, sin_half), dim=-1)
