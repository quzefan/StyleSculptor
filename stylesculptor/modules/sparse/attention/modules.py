from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder
from sklearn.decomposition import PCA
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    N, C = 1, 1024
    feat_var = feat.view(-1, N, C).var(dim=0, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(1, N, C)
    feat_mean = feat.view(-1, N, C).mean(dim=0).view(1, N, C)
    return feat_mean.view(1, C), feat_std.view(1, C)


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def find_cross_content_channels(k2_feats, k3_feats, num_channels_to_zero=512):
    """
    找到k2_feats和k3_feats之间最相似的num_channels_to_zero个通道，并将这些通道设为0。
    
    :param k2_feats: 输入特征张量，shape为(N, 1024)
    :param k3_feats: 输入特征张量，shape为(N, 1024)
    :param num_channels_to_zero: 需要置零的通道数，默认为512
    :return: 处理后的k2_feats
    """
    assert k2_feats.shape == k3_feats.shape, "k2_feats and k3_feats must have the same shape"
    N, C = k2_feats.shape  # N是patch数量，C是维度数
    
    # 计算每个通道的相关系数
    similarities = torch.zeros(C, device=k2_feats.device)
    for i in range(C):
        k2_channel = k2_feats[:, i]
        k3_channel = k3_feats[:, i]
        covariance = torch.mean((k2_channel - k2_channel.mean()) * (k3_channel - k3_channel.mean()))
        std_k2 = torch.std(k2_channel)
        std_k3 = torch.std(k3_channel)
        similarities[i] = covariance / (std_k2 * std_k3 + 1e-5)  # 避免除以零
    
    # 找到相关系数最高的num_channels_to_zero个通道
    _, topk_indices = torch.topk(similarities, num_channels_to_zero, largest=True)
    
    # # 将这些通道设为0
    # k2_feats_zeroed = k2_feats.clone()
    # k2_feats_zeroed[:, topk_indices] = 0
    
    return topk_indices

def find_one_content_channels(k2_feats, num_channels=512):
    # 计算每个通道的方差
    variances = torch.var(k2_feats, dim=0)
    # 找到方差最大的num_channels个通道
    _, max_variance_indices = torch.topk(variances, num_channels, largest=True)
    
    return max_variance_indices

def find_cross_style_channels(k2_feats, k3_feats, num_channels_to_zero=512):
    assert k2_feats.shape == k3_feats.shape, "k2_feats and k3_feats must have the same shape"
    N, C = k2_feats.shape 
    
    cos_similarity = torch.nn.CosineSimilarity(dim=0)
    similarities = torch.zeros(C, device=k2_feats.device)
    
    for i in range(C):
        similarities[i] = cos_similarity(k2_feats[:, i], k3_feats[:, i])
    
    _, topk_indices = torch.topk(similarities, num_channels_to_zero, largest=True)
    
    return topk_indices

def find_one_style_channels(k2_feats, num_channels=512):
    variances = torch.var(k2_feats, dim=0)
    _, min_variance_indices = torch.topk(variances, num_channels, largest=False)
    
    return min_variance_indices

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
        return qkv
    
    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                qkv = self._rope(qkv)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))

            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h
    
    def forward_Peserve(self, x0: Union[SparseTensor, torch.Tensor], x1: Union[SparseTensor, torch.Tensor], x2: Union[SparseTensor, torch.Tensor], x3: Union[SparseTensor, torch.Tensor], intensity: int, context: Optional[Union[SparseTensor, torch.Tensor]] = None) -> Union[SparseTensor, torch.Tensor]:       
        
        intensity_list = [1024, 750, 500, 350, 200, 0]
        qkv0 = self._linear(self.to_qkv, x0)
        qkv0 = self._fused_pre(qkv0, num_fused=3)
        
        qkv1 = self._linear(self.to_qkv, x1)
        qkv1 = self._fused_pre(qkv1, num_fused=3)
        
        qkv2 = self._linear(self.to_qkv, x2)
        qkv2 = self._fused_pre(qkv2, num_fused=3)
        
        qkv3 = self._linear(self.to_qkv, x3)
        qkv3 = self._fused_pre(qkv3, num_fused=3)
        
        
        if self.use_rope:
            qkv0 = self._rope(qkv0)
            qkv1 = self._rope(qkv1)
            qkv2 = self._rope(qkv2)
            qkv3 = self._rope(qkv3)
            
        if self.qk_rms_norm:
            q0, k0, v0 = qkv0.unbind(dim=1)
            q1, k1, v1 = qkv1.unbind(dim=1)
            q2, k2, v2 = qkv2.unbind(dim=1)
            q3, k3, v3 = qkv3.unbind(dim=1)
            
            q0 = self.q_rms_norm(q0)
            k0 = self.k_rms_norm(k0)
            q1 = self.q_rms_norm(q1)
            k1 = self.k_rms_norm(k1)
            q2 = self.q_rms_norm(q2)
            k2 = self.k_rms_norm(k2)
            q3 = self.q_rms_norm(q3)
            k3 = self.k_rms_norm(k3)
            
        if self.attn_mode == "full":
            kv1_size, kv2_size = k1.feats.size(), k2.feats.size()
            k0_feats, v0_feats = k0.feats.view(kv1_size[0], -1), v0.feats.view(kv1_size[0], -1)
            k1_feats, v1_feats = k1.feats.view(kv1_size[0], -1), v1.feats.view(kv1_size[0], -1)
            k2_feats, v2_feats = k2.feats.view(kv2_size[0], -1), v2.feats.view(kv2_size[0], -1)
            k3_feats, v3_feats = k3.feats.view(kv2_size[0], -1), v3.feats.view(kv2_size[0], -1)
            x2_feats, x3_feats = x2.feats.view(kv1_size[0], -1), x3.feats.view(kv1_size[0], -1)
            kv_ori = torch.cat([k2_feats.view(k2.feats.size()).unsqueeze(1), v2_feats.view(v2.feats.size()).unsqueeze(1)], dim=1).unsqueeze(0)
            
            h0 = sparse_scaled_dot_product_attention(q0, k0, v0)
            h2 = sparse_scaled_dot_product_attention(q2, k2, v2)
            h3 = sparse_scaled_dot_product_attention(q3, k3, v3)
            h2_feats, h3_feats = h2.feats.view(kv1_size[0], -1), h3.feats.view(kv1_size[0], -1)
            
            k3_feats_self_indices = find_one_content_channels(h3_feats, num_channels=intensity_list[intensity])
            mask_all = k3_feats_self_indices

            # random_mask = torch.randperm(k2_feats.size(1))[:len(mask_all)] 
            # random_mask = torch.randperm(k2_feats.size(1))[:200]     
             
            mask_iszero = torch.ones_like(k0_feats[0], dtype=torch.bool)
            mask_iszero[mask_all] = 0
            
            k2_feats_zeroed = k2_feats.clone()           
            v2_feats_zeroed = v2_feats.clone()
            v2_feats_zeroed[:, mask_all] = 0
            
            k0_feats_ones = k0_feats.clone()           
            v0_feats_ones = v0_feats.clone()
            v0_feats_ones[:, mask_iszero] = 0

            k1_feats_ones = k1_feats.clone()           
            v1_feats_ones = v1_feats.clone()
            v1_feats_ones[:, mask_iszero] = 0

            kv = torch.cat([k2_feats_zeroed.view(k2.feats.size()).unsqueeze(1), v2_feats_zeroed.view(v2.feats.size()).unsqueeze(1)], dim=1).unsqueeze(0)
            kv_content = torch.cat([k0_feats_ones.view(k0.feats.size()).unsqueeze(1), v0_feats_ones.view(v0.feats.size()).unsqueeze(1)], dim=1).unsqueeze(0)

            h1 = sparse_scaled_dot_product_attention(q1, kv) + sparse_scaled_dot_product_attention(q0, kv_content)

        elif self.attn_mode == "serialized":
            q1, k1, v1 = qkv1.unbind(dim=1)
            q2, k2, v2 = qkv2.unbind(dim=1)
            q1 = q1.replace(feats=adaptive_instance_normalization(q1.feats, q2.feats))
            k1 = k1.replace(feats=adaptive_instance_normalization(k1.feats, k2.feats))
            qkv = qkv1.replace(torch.stack([q1.feats, k1.feats, v1.feats], dim=1))
            h = sparse_serialized_scaled_dot_product_self_attention(
                qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
            )
        elif self.attn_mode == "windowed":
            q1, k1, v1 = qkv1.unbind(dim=1)
            q2, k2, v2 = qkv2.unbind(dim=1)
            q1 = q1.replace(feats=adaptive_instance_normalization(q1.feats, q2.feats))
            k1 = k1.replace(feats=adaptive_instance_normalization(k1.feats, k2.feats))
            qkv = qkv1.replace(torch.stack([q1.feats, k1.feats, v1.feats], dim=1))
            h = sparse_windowed_scaled_dot_product_self_attention(
                qkv, self.window_size, shift_window=self.shift_window
            )

        h0 = self._reshape_chs(h0, (-1,))
        h0 = self._linear(self.to_out, h0)
        
        h1 = self._reshape_chs(h1, (-1,))
        h1 = self._linear(self.to_out, h1)
        
        h2 = self._reshape_chs(h2, (-1,))
        h2 = self._linear(self.to_out, h2)
        
        h3 = self._reshape_chs(h3, (-1,))
        h3 = self._linear(self.to_out, h3)
        
        return h0, h1, h2, h3
    
    