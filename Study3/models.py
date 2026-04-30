
from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, GPT2Model


class FusionOp(str, Enum):
    WEIGHTED_SUM    = "weighted_sum"
    CONCAT          = "concat"
    CROSS_ATTENTION = "cross_attention"
    GATED           = "gated"


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class FusionLayer(nn.Module):
    """Configurable fusion of two (or three) pooled encoder outputs.

    For WEIGHTED_SUM / CONCAT / GATED the inputs are pooled vectors
    of shape (B, D_i). For CROSS_ATTENTION the module optionally takes
    full token sequences of shape (B, L_i, D_i) through `forward_seq`.
    """

    def __init__(self, dims: list, fusion_dim: int = 256,
                 op: FusionOp = FusionOp.WEIGHTED_SUM,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.op = FusionOp(op)
        self.dims = dims
        self.fusion_dim = fusion_dim

        if self.op in (FusionOp.WEIGHTED_SUM, FusionOp.GATED,
                       FusionOp.CROSS_ATTENTION):
            # Each input projected to a shared fusion_dim
            self.projs = nn.ModuleList([
                nn.Linear(d, fusion_dim) for d in dims
            ])

        if self.op == FusionOp.CONCAT:
            self.concat_proj = nn.Linear(sum(dims), fusion_dim)

        if self.op == FusionOp.GATED:
            # Gate over concatenated projected vectors produces one weight per input
            self.gate = nn.Linear(fusion_dim * len(dims), len(dims))

        if self.op == FusionOp.CROSS_ATTENTION:
            # Standard multi-head cross-attention block + residual + LayerNorm.
            # Q comes from the first input (BERT), K and V from the second.
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fusion_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            self.norm1 = nn.LayerNorm(fusion_dim)
            self.ff = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim),
            )
            self.norm2 = nn.LayerNorm(fusion_dim)

    # ---- pooled-vector forward (WEIGHTED_SUM, CONCAT, GATED) ----
    def forward(self, *hs: torch.Tensor) -> torch.Tensor:
        if self.op == FusionOp.WEIGHTED_SUM:
            # tanh(sum_i W_i h_i + b)
            out = self.projs[0](hs[0])
            for i in range(1, len(hs)):
                out = out + self.projs[i](hs[i])
            return torch.tanh(out)

        if self.op == FusionOp.CONCAT:
            # Concatenate then project — matches the chapter's *text*
            cat = torch.cat(hs, dim=-1)
            return torch.tanh(self.concat_proj(cat))

        if self.op == FusionOp.GATED:
            # Sigmoid gate mixes the projected inputs
            projected = [p(h) for p, h in zip(self.projs, hs)]
            gates = torch.softmax(
                self.gate(torch.cat(projected, dim=-1)), dim=-1
            )  # (B, n_inputs)
            out = 0
            for i, pr in enumerate(projected):
                out = out + gates[..., i:i + 1] * pr
            return torch.tanh(out)

        if self.op == FusionOp.CROSS_ATTENTION:
            # Degenerate: pooled vectors have L=1. Use forward_seq for real attention.
            projected = [p(h).unsqueeze(1) for p, h in zip(self.projs, hs)]
            out, _ = self.cross_attn(projected[0], projected[1], projected[1])
            out = self.norm1(out + projected[0])
            out = self.norm2(out + self.ff(out))
            return out.squeeze(1)

        raise ValueError(self.op)

    # ---- sequence-level forward (CROSS_ATTENTION) ----
    def forward_seq(self, seq_q: torch.Tensor, seq_kv: torch.Tensor,
                    mask_q: Optional[torch.Tensor] = None,
                    mask_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Real cross-attention: queries from seq_q attend over seq_kv.

        seq_q : (B, L_q, D_q) — e.g., BERT token sequence
        seq_kv: (B, L_k, D_k) — e.g., GPT-2 token sequence
        mask_*: (B, L_*) with 1 for real tokens, 0 for padding

        Returns a pooled (B, fusion_dim) vector.
        """
        assert self.op == FusionOp.CROSS_ATTENTION
        q = self.projs[0](seq_q)
        kv = self.projs[1](seq_kv)

        # nn.MultiheadAttention key_padding_mask: True = ignore
        kpm = (mask_kv == 0) if mask_kv is not None else None
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=kpm)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ff(x))

        # Mean-pool over valid query positions
        if mask_q is not None:
            m = mask_q.unsqueeze(-1).float()
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class BertClassifier(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.5,
                 model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, n_classes)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class GPT2Classifier(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.5,
                 model_name: str = "gpt2"):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        h = self.gpt2.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, n_classes)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # With left padding the last position is the final real token
        last = out.last_hidden_state[:, -1, :]
        return self.classifier(self.dropout(last))


# ----------------------------------------------------------------------------
# Fusion models
# ----------------------------------------------------------------------------

class BertGPT2Fusion(nn.Module):
    """Dual-stream fusion with configurable op.

    Pick any FusionOp. The thesis chapter's equation corresponds to
    WEIGHTED_SUM. If you want an actual attention mechanism, pick
    CROSS_ATTENTION — and use seq-level pooling by setting
    `use_sequence_cross_attn=True` in the constructor.
    """

    def __init__(self, n_classes: int, fusion_dim: int = 256,
                 dropout: float = 0.5,
                 fusion_op: FusionOp = FusionOp.WEIGHTED_SUM,
                 use_sequence_cross_attn: bool = False,
                 num_heads: int = 4,
                 bert_name: str = "bert-base-uncased",
                 gpt_name: str = "gpt2"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.gpt2 = GPT2Model.from_pretrained(gpt_name)
        h_b = self.bert.config.hidden_size
        h_g = self.gpt2.config.hidden_size

        self.fusion = FusionLayer(
            dims=[h_b, h_g], fusion_dim=fusion_dim,
            op=fusion_op, num_heads=num_heads, dropout=0.1,
        )
        self.use_sequence_cross_attn = (
            fusion_op == FusionOp.CROSS_ATTENTION and use_sequence_cross_attn
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask, **_):
        b = self.bert(input_ids=bert_ids, attention_mask=bert_mask)
        g = self.gpt2(input_ids=gpt_ids,  attention_mask=gpt_mask)

        if self.use_sequence_cross_attn:
            fused = self.fusion.forward_seq(
                b.last_hidden_state, g.last_hidden_state,
                mask_q=bert_mask, mask_kv=gpt_mask,
            )
        else:
            h_bert = b.last_hidden_state[:, 0, :]
            h_gpt  = g.last_hidden_state[:, -1, :]
            fused = self.fusion(h_bert, h_gpt)

        return self.classifier(self.dropout(fused))


class BertGPT2CNNFusion(nn.Module):
    """BERT + GPT-2 + CNN(TextCNN over BERT tokens).

    Three streams are projected into a shared fusion space:
      - BERT [CLS] pooled vector
      - CNN features over BERT token sequence (captures local n-grams)
      - GPT-2 last-token vector
    """

    def __init__(self, n_classes: int, fusion_dim: int = 256,
                 cnn_filters: int = 128, kernel_sizes=(2, 3, 5),
                 dropout: float = 0.5,
                 fusion_op: FusionOp = FusionOp.WEIGHTED_SUM,
                 bert_name: str = "bert-base-uncased",
                 gpt_name: str = "gpt2"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.gpt2 = GPT2Model.from_pretrained(gpt_name)
        h_b = self.bert.config.hidden_size
        h_g = self.gpt2.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv1d(h_b, cnn_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        cnn_out = cnn_filters * len(kernel_sizes)

        self.fusion = FusionLayer(
            dims=[h_b, cnn_out, h_g], fusion_dim=fusion_dim,
            op=fusion_op, dropout=0.1,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask, **_):
        b_tokens = self.bert(input_ids=bert_ids, attention_mask=bert_mask).last_hidden_state
        h_bert_cls = b_tokens[:, 0, :]

        # TextCNN over BERT tokens
        x = b_tokens.transpose(1, 2)  # (B, H, L)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(-1)
            pooled.append(p)
        h_cnn = torch.cat(pooled, dim=1)

        g_out = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask).last_hidden_state
        h_gpt = g_out[:, -1, :]

        fused = self.fusion(h_bert_cls, h_cnn, h_gpt)
        return self.classifier(self.dropout(fused))

