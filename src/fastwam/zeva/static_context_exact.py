"""Serving-equivalent offline static task contexts, including tied Top-K scores.

For main-observed cache generation, a large GEMM batch of query readouts can
produce slightly different BF16 hidden states or FP32 similarity scores than
the single-query online retriever. Near-tied prototypes can change the
identity of a neighbor despite similar scores. Match the online [1,D]
query and [1,N] scoring shapes, caching bank keys/values on the device.
"""

from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm


@torch.inference_mode()
def build_serving_exact_static_contexts(
    retriever,
    readouts: Tensor,
    episode_ids: Sequence[str],
    *,
    batch_size: int = 512,
    progress: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build what retriever.retrieve(x[None], exclude_episode_ids=[id]) serves.

    Avoid the repeated Python per-query construction of N_bank exclusion
    masks and repeated host-to-device bank transfers, without changing the
    original single-query projection, similarity, torch.topk tie order,
    softmax, or context-value aggregation.
    """
    if readouts.ndim != 2 or not readouts.is_floating_point():
        raise ValueError("readouts must be a floating-point [N,input_dim] tensor")
    if len(readouts) != len(episode_ids) or not len(episode_ids):
        raise ValueError("episode identity and readout counts must agree and be positive")
    if batch_size < 1:
        raise ValueError("static context batch_size must be positive")

    bank = retriever.bank
    device = next(retriever.head.parameters()).device
    bank_keys = bank._keys.to(device=device)
    bank_values = bank._values.to(device=device)
    # Loaded bank entries have canonical unique episode IDs. Reuse metadata
    # directly rather than repeatedly materializing bank.entries clones.
    episode_to_bank_index = {
        str(entry["episode_id"]): i for i, entry in enumerate(bank._entries)
    }
    if len(episode_to_bank_index) != len(bank):
        raise ValueError("behavior bank episode identities must be unique")
    top_k = min(int(retriever.top_k), len(bank) - 1)
    if top_k < 1:
        raise ValueError("leave-one-out retrieval needs >=2 bank entries")

    num_queries = len(readouts)
    contexts = torch.empty((num_queries, bank.value_dim), dtype=torch.float32)
    indices = torch.empty((num_queries, top_k), dtype=torch.int64)
    scores = torch.empty((num_queries, top_k), dtype=torch.float32)
    iterator = range(0, num_queries, batch_size)
    if progress:
        iterator = tqdm(
            iterator, total=(num_queries + batch_size - 1) // batch_size,
            desc="static context (serving-exact)", unit="batch",
        )
    main_observed = bank.metadata.get("profile") == "main_observed"
    for start in iterator:
        end = min(start + batch_size, num_queries)
        context_buffer: list[Tensor] = []
        index_buffer: list[Tensor] = []
        score_buffer: list[Tensor] = []
        for index in range(start, end):
            # Main serving computes B=1 under FP32 weights + BF16 autocast.
            q = retriever.query(readouts[index:index + 1])
            autocast_off = (
                torch.autocast(device_type=q.device.type, enabled=False)
                if main_observed else nullcontext()
            )
            with autocast_off:
                query = torch.as_tensor(q).float()
                similarity = F.normalize(query, dim=-1) @ bank_keys.T
                own_index = episode_to_bank_index.get(str(episode_ids[index]))
                if own_index is not None:
                    similarity[0, own_index] = -torch.inf
                found_scores, found_indices = similarity.topk(
                    top_k, dim=-1, largest=True, sorted=True
                )
                weights = F.softmax(found_scores / bank.temperature, dim=-1)
                selected = bank_values[found_indices]
                found_context = torch.einsum("bk,bkd->bd", weights, selected)
            context_buffer.append(found_context)
            index_buffer.append(found_indices)
            score_buffer.append(found_scores)
        contexts[start:end] = torch.cat(context_buffer, dim=0).float().cpu()
        indices[start:end] = torch.cat(index_buffer, dim=0).cpu()
        scores[start:end] = torch.cat(score_buffer, dim=0).float().cpu()
    if not bool(torch.isfinite(contexts).all() and torch.isfinite(scores).all()):
        raise FloatingPointError("offline static context contains NaN/Inf")
    return contexts, indices, scores
