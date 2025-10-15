from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
    
    def _forward_CrossAtten_Peserve(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, mod: torch.Tensor, context1: torch.Tensor, context2: torch.Tensor, context3: torch.Tensor, intensity: int):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        
        h0 = self.norm1(x0)
        h0 = h0 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        h1 = self.norm1(x1)
        h1 = h1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        h2 = self.norm1(x2)
        h2 = h2 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        h3 = self.norm1(x3)
        h3 = h3 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        h0, h1, h2, h3 = self.self_attn.forward_CrossAtten_Preserve(h0, h1, h2, h3, intensity)

        h0 = h0 * gate_msa.unsqueeze(1)
        x0 = x0 + h0
        h0 = self.norm2(x0)
        h0 = self.cross_attn(h0, context1)
        x0 = x0 + h0
        h0 = self.norm3(x0)
        h0 = h0 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h0 = self.mlp(h0)
        h0 = h0 * gate_mlp.unsqueeze(1)
        x0 = x0 + h0
        
        h1 = h1 * gate_msa.unsqueeze(1)
        x1 = x1 + h1
        h1 = self.norm2(x1)
        h1 = self.cross_attn(h1, context1)
        x1 = x1 + h1
        h1 = self.norm3(x1)
        h1 = h1 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h1 = self.mlp(h1)
        h1 = h1 * gate_mlp.unsqueeze(1)
        x1 = x1 + h1
        
        h2 = h2 * gate_msa.unsqueeze(1)
        x2 = x2 + h2
        h2 = self.norm2(x2)
        h2 = self.cross_attn(h2, context2)
        x2 = x2 + h2
        h2 = self.norm3(x2)
        h2 = h2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h2 = self.mlp(h2)
        h2 = h2 * gate_mlp.unsqueeze(1)
        x2 = x2 + h2
        
        h3 = h3 * gate_msa.unsqueeze(1)
        x3 = x3 + h3
        h3 = self.norm2(x3)
        h3 = self.cross_attn(h3, context3)
        x3 = x3 + h3
        h3 = self.norm3(x3)
        h3 = h3 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h3 = self.mlp(h3)
        h3 = h3 * gate_mlp.unsqueeze(1)
        x3 = x3 + h3
        
        return x0, x1, x2, x3
    
    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
        