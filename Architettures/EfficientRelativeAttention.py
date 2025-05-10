import math
import torch
from torch import nn


class EfficientRelativeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must divide num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Standard QKV projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Learnable relative embeddings of shape (2*max_len-1, head_dim)
        self.max_len = max_len
        self.rel_k = nn.Parameter(torch.randn(2 * max_len - 1, self.head_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x    : (B, L, D)
        mask : (B, L) or (B, 1, L, L) optional
        """
        B, L, _ = x.shape
        assert L <= self.max_len, "Sequence length exceeds max_len"

        # 1) Project to QKV and reshape to (B, H, L, head_dim)
        qkv = self.qkv_proj(x)  # (B, L, 3D)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, H, L, D_k)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 2) Content-based scores: Q @ K^T → (B, H, L, L)
        content_scores = torch.matmul(q, k.transpose(-2, -1))

        # 3) Relative position logits
        rel_logits = relative_logits_1d(q, self.rel_k[self.max_len - L: self.max_len + L - 1])

        # 4) Combine & scale
        scores = (content_scores + rel_logits) * self.scale

        # 5) Mask (if provided) and softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        # 6) Weighted sum over V
        out = torch.matmul(attn, v)  # (B, H, L, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        return self.out_proj(out)
