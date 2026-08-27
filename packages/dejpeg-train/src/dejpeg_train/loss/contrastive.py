"""Degradation-encoder losses (spec §3, Phase 1).

MoCo-style contrastive (1.0) + LQ reconstruction (0.5) + auxiliary QF regression
(0.1). This is the deg-encoder's own training objective; the resulting embedding
conditions the restore net. Contrastive uses a queue of negative keys (MoCo).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(
    query: torch.Tensor, key: torch.Tensor, queue: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """query (N,D) [online], key (N,D) [momentum positive], queue (K,D) [negatives]."""
    q = F.normalize(query, dim=1)
    k = F.normalize(key, dim=1)
    queue = F.normalize(queue, dim=1)
    l_pos = torch.einsum("nd,nd->n", q, k).unsqueeze(1)            # (N,1)
    l_neg = torch.einsum("nd,kd->nk", q, queue)                    # (N,K)
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def enqueue_dequeue(queue: torch.Tensor, keys: torch.Tensor, max_size: int) -> torch.Tensor:
    """Rolling FIFO queue of momentum keys."""
    keys = F.normalize(keys, dim=1).detach()
    new = torch.cat([queue, keys], dim=0)
    return new[-max_size:]


class DegEncoderLoss(nn.Module):
    """Combines contrastive + LQ reconstruction + QF regression.

    The recon head and qf head are small and live here so the deg-encoder stays a
    pure feature extractor. Contrastive weight 1.0, recon 0.5, qf-reg 0.1.
    """

    def __init__(self, emb_dim: int = 32, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.recon_head = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.GELU(),
            nn.Linear(256, 3 * 64 * 64), nn.Sigmoid(),
        )
        self.qf_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        queue: torch.Tensor,
        lq_target: torch.Tensor,
        true_qf: torch.Tensor,
        w_contrast: float = 1.0,
        w_recon: float = 0.5,
        w_qf: float = 0.1,
    ):
        l_c = info_nce(query, key, queue, self.temperature)
        lq_pred = self.recon_head(query).view(query.shape[0], 3, 64, 64)
        l_r = F.mse_loss(lq_pred, lq_target)
        qf_pred = self.qf_head(query).squeeze(-1)
        l_q = F.l1_loss(qf_pred, true_qf.float().squeeze(-1))
        return w_contrast * l_c + w_recon * l_r + w_qf * l_q, {
            "contrastive": l_c.detach(),
            "recon": l_r.detach(),
            "qf": l_q.detach(),
        }
