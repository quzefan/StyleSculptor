from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention

def find_and_one_similar_channels(k2_feats, num_channels=512):

    variances = torch.var(k2_feats, dim=0)
    
    _, min_variance_indices = torch.topk(variances, num_channels, largest=True)
    
    return min_variance_indices

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
    

    def forward_CrossAtten_Preserve(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, intensity: int, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x1.shape
        intensity_list = [1024, 1024, 1000, 975, 950, 925]
        qkv0 = self.to_qkv(x0)
        qkv1 = self.to_qkv(x1)
        qkv2 = self.to_qkv(x2)
        qkv3 = self.to_qkv(x3)
        qkv0, qkv1, qkv2, qkv3 = qkv0.reshape(B, L, 3, self.num_heads, -1), qkv1.reshape(B, L, 3, self.num_heads, -1), qkv2.reshape(B, L, 3, self.num_heads, -1), qkv3.reshape(B, L, 3, self.num_heads, -1)
        if self.use_rope:
            q0, k0, v0 = qkv0.unbind(dim=2)
            q0, k0 = self.rope(q0, k0, indices)
            q1, k1, v1 = qkv1.unbind(dim=2)
            q1, k1 = self.rope(q1, k1, indices)
            q2, k2, v2 = qkv2.unbind(dim=2)
            q2, k2 = self.rope(q2, k2, indices)
            qkv = torch.stack([q1, k2, v2], dim=2)
        if self.attn_mode == "full":
            if self.qk_rms_norm:
                q0, k0, v0 = qkv0.unbind(dim=2)
                q1, k1, v1 = qkv1.unbind(dim=2)
                q2, k2, v2 = qkv2.unbind(dim=2)
                q3, k3, v3 = qkv3.unbind(dim=2)

                q0 = self.q_rms_norm(q0)
                k0 = self.k_rms_norm(k0)
                q1 = self.q_rms_norm(q1)
                k1 = self.k_rms_norm(k1)
                q2 = self.q_rms_norm(q2)
                k2 = self.k_rms_norm(k2)
                q3 = self.q_rms_norm(q3)
                k3 = self.k_rms_norm(k3)
                
                kv_size = k1.size()
                k0_feats, v0_feats = k0.view(kv_size[1], -1), v0.view(kv_size[1], -1)
                k1_feats, v1_feats = k1.view(kv_size[1], -1), v1.view(kv_size[1], -1)
                k2_feats, v2_feats = k2.view(kv_size[1], -1), v2.view(kv_size[1], -1)
                k3_feats, v3_feats = k3.view(kv_size[1], -1), v3.view(kv_size[1], -1)
                
                h0 = scaled_dot_product_attention(q0, k0, v0)

                h2 = scaled_dot_product_attention(q2, k2, v2)
                h3 = scaled_dot_product_attention(q3, k3, v3)
                h2_feats, h3_feats = h2.view(kv_size[1], -1), h3.view(kv_size[1], -1)
                
                k3_feats_self_indices = find_and_one_similar_channels(h3_feats, num_channels=intensity_list[intensity])
                mask_all = k3_feats_self_indices

                # random_mask = torch.randperm(k2_feats.size(1))[:len(mask_all)]

                
                mask_iszero = torch.ones_like(k0_feats[0], dtype=torch.bool)
                mask_iszero[mask_all] = 0
                
                k2_feats_zeroed = k2_feats.clone()
                
                v2_feats_zeroed = v2_feats.clone()
                v2_feats_zeroed[:, mask_all] = 0
                
                k0_feats_ones = k0_feats.clone()
                
                v0_feats_ones = v0_feats.clone()
                v0_feats_ones[:, mask_iszero] = 0

                kv_style = torch.cat([k2_feats_zeroed.view(k2.size()).unsqueeze(2), v2_feats_zeroed.view(v2.size()).unsqueeze(2)], dim=2)
                kv_content = torch.cat([k0_feats_ones.view(k0.size()).unsqueeze(2), v0_feats_ones.view(v0.size()).unsqueeze(2)], dim=2)

                h1 = scaled_dot_product_attention(q1, kv_style) + scaled_dot_product_attention(q0, kv_content)

            else:
                q1, k1, v1 = qkv1.unbind(dim=2)
                q2, k2, v2 = qkv2.unbind(dim=2)
                k, v = torch.cat([k1, k2], dim=1), torch.cat([v1, v2], dim=1)
                kv = torch.cat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
                # qkv = torch.stack([q1, k2, v2], dim=2)
                h = scaled_dot_product_attention(q1, kv)
        elif self.attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        h0 = h0.reshape(B, L, -1)
        h0 = self.to_out(h0)
        
        h1 = h1.reshape(B, L, -1)
        h1 = self.to_out(h1)
        
        h2 = h2.reshape(B, L, -1)
        h2 = self.to_out(h2)
        
        h3 = h3.reshape(B, L, -1)
        h3 = self.to_out(h3)
        return h0, h1, h2, h3
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h
