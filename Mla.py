import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Optional
from MLAkernels import _attention


class MultiHeadAttention(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int,
            head_dim: int,
            bias: bool = True, 
            dropout: float = 0.0
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim 

        self.Q_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.K_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.V_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.O_proj = nn.Linear(embed_dim, embed_dim, bias = bias)

    def forward(
            self, 
            Query : torch.Tensor,
            Key: torch.Tensor,
            Value : torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    )->torch.Tensor:
        
        batch_size, tgt_len, _ = Query.shape
        src_len = Key.shape[1]

        q = self.Q_proj(Query)  
        k = self.K_proj(Key)   
        v = self.V_proj(Value)

        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = _attention.apply(q, k, v, is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, self.embed_dim
        )
        attn_output = self.O_proj(attn_output)
        return attn_output



        