"""Static episode context: match the actual single-query retriever.

Training-time vectorization may change BF16 head rounding or a tied top-K
order. The formal cache must be bit-identical to online single-query serving.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from fastwam.zeva.static_context_exact import build_serving_exact_static_contexts


class _Bank:
    def __init__(self, keys, values, ids):
        self._keys = F.normalize(keys, dim=-1)
        self._values = values
        self._entries = [{"episode_id": identity} for identity in ids]
        self.metadata = {"profile": "main_observed"}
        self.value_dim = values.shape[1]
        self.temperature = 0.07

    def __len__(self):
        return len(self._entries)

    def retrieve(self, query, *, top_k, exclude_episode_ids):
        q = torch.as_tensor(query).float()
        scores_all = F.normalize(q, dim=-1) @ self._keys.T
        eligible = torch.tensor([
            [str(entry.get("episode_id", "")) != episode_id for entry in self._entries]
            for episode_id in exclude_episode_ids
        ], device=q.device, dtype=torch.bool)
        scores_all = scores_all.masked_fill(~eligible, -torch.inf)
        scores, indices = scores_all.topk(top_k, dim=-1, largest=True, sorted=True)
        weights = F.softmax(scores / self.temperature, dim=-1)
        selected = self._values[indices]
        return SimpleNamespace(
            values=torch.einsum("bk,bkd->bd", weights, selected),
            indices=indices, scores=scores,
        )


class _Retriever:
    def __init__(self, bank, top_k, bump_multi=0):
        self.bank = bank
        self.top_k = top_k
        self.head = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(4))
        self.bump_multi = bump_multi
        self.head.eval()

    def query(self, readout):
        q = self.head(readout)
        if len(readout) > 1:
            q = q + self.bump_multi
        return F.normalize(q, dim=-1)

    def retrieve(self, readout, exclude_episode_ids):
        q = self.query(readout)
        return self.bank.retrieve(
            q, top_k=min(self.top_k, len(self.bank) - 1),
            exclude_episode_ids=exclude_episode_ids,
        )


@pytest.mark.parametrize("count", [4, 14])
def test_single_query_matches_online_for_train_holdout_and_tied_keys(count):
    torch.manual_seed(count)
    keys = torch.randn(count, 4)
    keys[2] = keys[3]  # Exact tie: torch.topk order is authoritative.
    values = torch.randn(count, 7)
    ids = [f"ep-{i}" for i in range(count)]
    retriever = _Retriever(_Bank(keys, values, ids), top_k=2)
    readouts = torch.randn(count + 3, 4)
    query_ids = ids + ["heldout-a", "heldout-b", "heldout-c"]

    contexts, indices, scores = build_serving_exact_static_contexts(
        retriever, readouts, query_ids, batch_size=5, progress=False,
    )
    for i, episode_id in enumerate(query_ids):
        expected = retriever.retrieve(readouts[i:i + 1], exclude_episode_ids=[episode_id])
        torch.testing.assert_close(indices[i], expected.indices[0], rtol=0, atol=0)
        torch.testing.assert_close(scores[i], expected.scores[0], rtol=0, atol=0)
        torch.testing.assert_close(contexts[i], expected.values[0], rtol=0, atol=0)
    for i in range(count):
        assert i not in indices[i]


def test_batch_dependent_query_cannot_corrupt_serving_exact_cache():
    keys = torch.tensor([[1., 0, 0, 0], [1, 0, 0, .0001],
                         [-1, 0, 0, 0], [0, 1, 0, 0]])
    values = torch.arange(12).reshape(4, 3).float()
    retriever = _Retriever(
        _Bank(keys, values, ["a", "b", "c", "d"]),
        top_k=2, bump_multi=.1,
    )
    readouts = torch.tensor([[1., 0, 0, 0], [1, 0, 0, .001], [0, 1., 0., 0.]])
    assert not torch.equal(
        retriever.query(readouts),
        torch.cat([retriever.query(row[None]) for row in readouts]),
    )
    contexts, indices, scores = build_serving_exact_static_contexts(
        retriever, readouts, ["held0", "held1", "held2"],
        batch_size=3, progress=False,
    )
    for i in range(3):
        expected = retriever.retrieve(
            readouts[i:i + 1], exclude_episode_ids=[f"held{i}"]
        )
        torch.testing.assert_close(contexts[i], expected.values[0], rtol=0, atol=0)
        torch.testing.assert_close(indices[i], expected.indices[0], rtol=0, atol=0)
        torch.testing.assert_close(scores[i], expected.scores[0], rtol=0, atol=0)


def test_bad_episode_identity_and_batch_configuration_fail_early():
    bank = _Bank(torch.randn(3, 4), torch.randn(3, 7), ["a", "b", "c"])
    retriever = _Retriever(bank, 2)
    with pytest.raises(ValueError):
        build_serving_exact_static_contexts(
            retriever, torch.randn(3, 4), ["a"], progress=False
        )
    with pytest.raises(ValueError):
        build_serving_exact_static_contexts(
            retriever, torch.randn(3, 4), ["a", "b", "c"], batch_size=0
        )
