"""Frozen task-language classification followed by task-scoped CTE retrieval.

The selection order follows Zeva-Ego's TaskLanguageMemory: predict a task
from instruction-only input embeddings, use that task's mean CTE key, then
aggregate its top-five demonstrations. No image, online phase, episode ID or
ground-truth task label enters the classifier at inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint import checkpoint_sha256
from .task_context import TaskContextBank, TaskContextRetrievalResult


TASK_LANGUAGE_FEATURE_KIND = "frozen_task_input_embedding_mean_v1"
FEATURE_FORMAT = "fastwam_task_language_features_v1"
CLASSIFIER_FORMAT = "fastwam_task_language_classifier_v1"
ENCODER_FORMAT = "fastwam_frozen_task_language_encoder_v1"
PROMPT_NORMALIZATION = "zeva_task_only_strip_underscore_newline_v1"


def normalize_task_instruction(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Task-language queries require a nonempty raw instruction")
    return instruction.strip().replace("_", " ").replace("\n", " ") + "\n"


def _validate_table(table: Tensor) -> None:
    if not isinstance(table, Tensor) or table.ndim != 2 or min(table.shape) < 1 or not table.is_floating_point():
        raise ValueError("Frozen input embedding table must be a floating [vocabulary,feature] tensor")
    for block in table.split(1024):
        if not bool(torch.isfinite(block).all()):
            raise ValueError("Frozen input embedding table contains non-finite values")


def _tokenizer_digest(files: dict[str, bytes]) -> str:
    if not files:
        raise ValueError("A portable task-language encoder needs tokenizer snapshot files")
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or not name or not isinstance(content, bytes)):
            raise ValueError("Invalid tokenizer snapshot path/content")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


class FrozenTaskLanguageEncoder:
    """CPU-resident frozen input table; only pooled features move to a device.

    This deliberately does not call the contextual text encoder or any policy
    backbone. Its portable snapshot is independent of Stage2B foundation updates.
    """

    def __init__(self, embedding_table: Tensor, tokenizer, *, max_length: int = 200,
                 metadata: dict | None = None, copy_table: bool = True):
        _validate_table(embedding_table)
        if max_length < 1:
            raise ValueError("Task-language maximum length must be positive")
        table = embedding_table.detach().cpu().contiguous()
        self.embedding_table = table.clone() if copy_table else table
        self.embedding_table.requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length, self.metadata = int(max_length), dict(metadata or {})
        self.feature_dim = int(table.shape[1])
        self._tokenizer_directory = None

    @torch.no_grad()
    def encode(self, instructions: str | list[str], *, device=None) -> Tensor:
        instructions = [instructions] if isinstance(instructions, str) else list(instructions)
        if not instructions:
            raise ValueError("Task-language query batch must not be empty")
        prompts = [normalize_task_instruction(value) for value in instructions]
        encoded = self.tokenizer(prompts, max_length=self.max_length, padding="max_length",
                                 truncation=True, return_tensors="pt")
        ids = torch.as_tensor(encoded["input_ids"], device="cpu").long()
        mask = torch.as_tensor(encoded["attention_mask"], device="cpu").bool()
        if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] != len(prompts):
            raise ValueError("Tokenizer must provide aligned [batch,length] IDs and attention masks")
        if not bool(mask.any(-1).all()) or bool((ids < 0).any()) or bool((ids >= len(self.embedding_table)).any()):
            raise ValueError("Task-language tokenizer returned empty masks or out-of-vocabulary IDs")
        embedded = F.embedding(ids, self.embedding_table).float()
        features = embedded.masked_fill(~mask.unsqueeze(-1), 0).sum(1) / mask.sum(1, keepdim=True)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Non-finite task-language pooled features")
        return features if device is None else features.to(device=device)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenTaskLanguageEncoder":
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if (payload.get("format") != ENCODER_FORMAT or payload.get("feature_kind") != TASK_LANGUAGE_FEATURE_KIND
                or payload.get("prompt_normalization") != PROMPT_NORMALIZATION):
            raise ValueError("Unsupported frozen task-language encoder semantics")
        files = payload.get("tokenizer_files", {})
        if _tokenizer_digest(files) != payload.get("tokenizer_sha256"):
            raise ValueError("Task-language tokenizer snapshot identity mismatch")
        from transformers import AutoTokenizer

        temporary = tempfile.TemporaryDirectory(prefix="iclwam_task_tokenizer_")
        try:
            for name, content in files.items():
                target = Path(temporary.name) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            tokenizer = AutoTokenizer.from_pretrained(temporary.name, local_files_only=True, trust_remote_code=False)
            result = cls(payload["embedding_table"], tokenizer, max_length=int(payload["max_length"]),
                         metadata=payload.get("metadata"), copy_table=False)
            result._tokenizer_directory = temporary
            result.artifact_path = str(Path(path).expanduser().resolve())
            result.artifact_sha256 = checkpoint_sha256(path)
            return result
        except BaseException:
            temporary.cleanup()
            raise


def export_task_language_encoder(path: str | Path, embedding_table: Tensor, tokenizer, *,
                                 metadata: dict | None = None, max_length: int = 200) -> dict:
    """Snapshot an input embedding table and its actual tokenizer together."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError("Task-language encoder export requires an unused output path")
    _validate_table(embedding_table)
    if max_length < 1:
        raise ValueError("Task-language maximum length must be positive")
    # FastWAM wraps the Hugging Face tokenizer for contextual policy encoding;
    # task-language encoding uses the underlying tokenizer and explicit prompts.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    with tempfile.TemporaryDirectory(prefix="iclwam_task_tokenizer_export_") as folder:
        tokenizer.save_pretrained(folder)
        files = {str(file.relative_to(folder)): file.read_bytes() for file in Path(folder).rglob("*") if file.is_file()}
    payload = {
        "format": ENCODER_FORMAT, "feature_kind": TASK_LANGUAGE_FEATURE_KIND,
        "prompt_normalization": PROMPT_NORMALIZATION, "max_length": int(max_length),
        "embedding_table": embedding_table.detach().cpu().contiguous(),
        "tokenizer_files": files, "tokenizer_sha256": _tokenizer_digest(files), "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"feature_dim": int(embedding_table.shape[1]), "vocabulary_size": len(embedding_table),
            "encoder_sha256": checkpoint_sha256(path), "tokenizer_sha256": payload["tokenizer_sha256"]}


def export_task_language_encoder_from_fastwam_config(model_cfg, path: str | Path, *,
                                                    metadata: dict | None = None, max_length: int = 200) -> dict:
    """Load only the frozen Wan input table/tokenizer, never DiT/VAE/T5 blocks."""
    from fastwam.models.wan22.helpers.loader import _resolve_configs
    from transformers import AutoTokenizer

    _, text_config, _, tokenizer_config = _resolve_configs(
        str(model_cfg["model_id"]), str(model_cfg["tokenizer_model_id"]),
        bool(model_cfg.get("redirect_common_files", True)),
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    if not isinstance(text_config.path, str) or not Path(text_config.path).is_file():
        raise ValueError("Task-language export requires one resolved text encoder checkpoint")
    source = Path(text_config.path)
    if source.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(source, framework="pt", device="cpu") as handle:
            if "token_embedding.weight" not in handle.keys():
                raise ValueError("Wan text checkpoint does not contain token_embedding.weight")
            table = handle.get_tensor("token_embedding.weight")
    else:
        try:
            state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            state = torch.load(source, map_location="cpu", weights_only=True)
        for wrapper in ("state_dict", "module", "model_state"):
            if len(state) == 1 and wrapper in state:
                state = state[wrapper]
        if "token_embedding.weight" not in state:
            raise ValueError("Wan text checkpoint does not contain token_embedding.weight")
        table = state["token_embedding.weight"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_config.path), local_files_only=True, trust_remote_code=False)
    return export_task_language_encoder(
        path, table, tokenizer, max_length=max_length,
        metadata={**dict(metadata or {}), "source_text_checkpoint_sha256": checkpoint_sha256(source),
                  "source_model_id": str(model_cfg["model_id"]),
                  "tokenizer_model_id": str(model_cfg["tokenizer_model_id"]), "source_tensor": "token_embedding.weight"},
    )


@dataclass(frozen=True)
class TaskLanguageHeadConfig:
    input_dim: int = 4096
    hidden_dim: int = 1024
    output_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.input_dim, self.hidden_dim, self.output_dim) < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid task-language classifier dimensions or dropout")

    def as_dict(self):
        return asdict(self)


class TaskLanguageHead(nn.Module):
    def __init__(self, config: TaskLanguageHeadConfig):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim),
                                     nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, config.output_dim))

    def forward(self, features: Tensor) -> Tensor:
        if (features.ndim != 2 or features.shape[-1] != self.config.input_dim
                or not bool(torch.isfinite(features).all())):
            raise ValueError(f"Task-language features must be finite [batch,{self.config.input_dim}]")
        projected = self.network(features.float())
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("Task-language classifier produced non-finite task scores")
        return F.normalize(projected, dim=-1)


def validate_task_language_bank(bank: TaskContextBank, expected: dict | None = None) -> None:
    for key, value in {"split": "train", "readout_kind": TASK_LANGUAGE_FEATURE_KIND,
                       "memory_semantics": "predicted_effect_trace", **dict(expected or {})}.items():
        if bank.metadata.get(key) != value:
            raise ValueError(f"Task-language bank identity mismatch: {key}")
    entries = bank.entries
    ids = [entry.get("episode_id") for entry in entries]
    if (any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids)
            or any(not str(entry.get("task_id", "")) or entry.get("split") != "train" for entry in entries)):
        raise ValueError("Task-language memory needs unique train-only episode IDs and declared tasks")
    if bank.temperature != 1.0:
        raise ValueError("Task-language task-scoped top-K aggregation uses reference temperature 1.0")


class TaskLanguageMemory(nn.Module):
    """Frozen language classifier -> mean task key -> same-task top-K values."""

    def __init__(self, bank: TaskContextBank, head: TaskLanguageHead, task_names: list[str] | tuple[str, ...],
                 task_prototypes: Tensor, *, top_k: int = 5, encoder: FrozenTaskLanguageEncoder | None = None):
        super().__init__()
        validate_task_language_bank(bank)
        names = tuple(task_names)
        if not names or any(not isinstance(value, str) or not value for value in names) or len(set(names)) != len(names):
            raise ValueError("Classifier task names must be nonempty and unique")
        if (task_prototypes.shape != (len(names), head.config.output_dim)
                or not bool(torch.isfinite(task_prototypes).all()) or bool((task_prototypes.norm(dim=-1) <= 0).any())):
            raise ValueError("Classifier requires one finite nonzero prototype per declared task")
        if top_k < 1:
            raise ValueError("Task-language retrieval top_k must be positive")
        bank_tasks = sorted({str(entry["task_id"]) for entry in bank.entries})
        if not set(bank_tasks).issubset(names):
            raise ValueError("All declared bank tasks must be represented by the frozen language classifier")
        self.bank, self.head, self.encoder = bank, head, encoder
        self.task_names, self.bank_task_names, self.top_k = names, tuple(bank_tasks), min(int(top_k), len(bank))
        self.register_buffer("language_prototypes", F.normalize(task_prototypes.detach().float(), dim=-1))
        self.register_buffer("task_mapping", torch.tensor([bank_tasks.index(name) if name in bank_tasks else -1 for name in names]))
        self.register_buffer("keys", torch.stack([entry["retrieval_key"] for entry in bank.entries]).float())
        self.register_buffer("values", torch.stack([entry["behavior_value"] for entry in bank.entries]).float())
        self.register_buffer("entry_task", torch.tensor([bank_tasks.index(str(entry["task_id"])) for entry in bank.entries]))
        self.episode_ids = tuple(entry["episode_id"] for entry in bank.entries)
        self.last_diagnostics = []
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        # A surrounding policy.train() must never unfreeze stochastic retrieval.
        return super().train(False)

    @torch.no_grad()
    def retrieve(self, features: Tensor, *, exclude_episode_ids=None) -> TaskContextRetrievalResult:
        parameter = next(self.head.parameters())
        features = features.to(device=parameter.device, dtype=torch.float32)
        scores = self.head(features) @ self.language_prototypes.T
        scores = scores.masked_fill(self.task_mapping[None] < 0, -torch.inf)
        confidence, predicted = scores.max(-1)
        tasks = self.task_mapping[predicted]
        eligible = self.entry_task[None] == tasks[:, None]
        if exclude_episode_ids is not None:
            excluded = list(exclude_episode_ids)
            if len(excluded) != len(features):
                raise ValueError("One optional excluded episode ID is required per query")
            eligible &= torch.tensor([[identity is None or stored != identity for stored in self.episode_ids]
                                      for identity in excluded], device=eligible.device)
        if not bool(eligible.any(-1).all()):
            raise ValueError("Language-predicted task has no eligible training demonstrations after exclusion")
        # Leave-one-episode-out affects both the task mean query and support.
        query = F.normalize(eligible.float() @ self.keys / eligible.sum(-1, keepdim=True), dim=-1)
        all_scores = (query @ F.normalize(self.keys, dim=-1).T).masked_fill(~eligible, -torch.inf)
        selected_scores, indices = all_scores.topk(self.top_k, dim=-1)
        valid = torch.isfinite(selected_scores)
        weights = F.softmax(selected_scores, dim=-1)  # reference has no 0.07 temperature here
        output = (weights.unsqueeze(-1) * self.values[indices]).sum(1)
        entries = self.bank.entries
        self.last_diagnostics, sources = [], []
        for row in range(len(features)):
            diagnostic = {"task": self.task_names[int(predicted[row])], "score": float(confidence[row]),
                          "support_count": int(eligible[row].sum())}
            self.last_diagnostics.append(diagnostic)
            sources.append({"predicted_task": diagnostic["task"], "language_score": diagnostic["score"],
                            "top_k": [{key: value for key, value in entries[int(index)].items()
                                       if key not in {"retrieval_key", "behavior_value"}}
                                      for index in indices[row][valid[row]].tolist()]})
        return TaskContextRetrievalResult(output, indices.masked_fill(~valid, -1), selected_scores, tuple(sources))

    def forward(self, features: Tensor) -> Tensor:
        return self.retrieve(features).values

    @classmethod
    def load(cls, bank_path, head_path, *, encoder_path, top_k=5, device="cpu", expected=None):
        bank = TaskContextBank.load(bank_path)
        validate_task_language_bank(bank, expected)
        payload = torch.load(head_path, map_location="cpu", weights_only=False)
        if (payload.get("format") != CLASSIFIER_FORMAT or payload.get("source_feature") != "task_language"
                or payload.get("feature_kind") != TASK_LANGUAGE_FEATURE_KIND or int(payload.get("step", 0)) < 1):
            raise ValueError("Task-language retrieval requires a trained task-language classifier checkpoint")
        if payload.get("bank_sha256") != checkpoint_sha256(bank_path):
            raise ValueError("Task-language classifier/bank identity mismatch")
        if payload.get("encoder_sha256") != checkpoint_sha256(encoder_path):
            raise ValueError("Task-language classifier/encoder identity mismatch")
        if bank.metadata.get("encoder_sha256", payload["encoder_sha256"]) != payload["encoder_sha256"]:
            raise ValueError("Task-language bank/encoder identity mismatch")
        train, validation = set(payload.get("train_episode_ids", [])), set(payload.get("validation_episode_ids", []))
        bank_ids = {entry["episode_id"] for entry in bank.entries}
        if not train or not validation or train & validation or train | validation != bank_ids:
            raise ValueError("Task-language classifier training/selection must stay inside the CTE train split")
        encoder = FrozenTaskLanguageEncoder.load(encoder_path)
        head = TaskLanguageHead(TaskLanguageHeadConfig(**payload["config"]))
        if head.config.input_dim != encoder.feature_dim:
            raise ValueError("Task-language classifier/encoder dimensions differ")
        head.load_state_dict(payload["model"], strict=True)
        result = cls(bank, head, payload["task_names"], payload["task_prototypes"], top_k=top_k, encoder=encoder)
        result.to(device=device)
        result.classifier_payload = payload
        return result


def task_language_identity(bank_path, head_path, encoder_path, top_k=5) -> dict:
    return {"mode": "task_language", "bank_sha256": checkpoint_sha256(bank_path),
            "head_sha256": checkpoint_sha256(head_path), "language_encoder_sha256": checkpoint_sha256(encoder_path),
            "top_k": int(top_k), "readout_kind": TASK_LANGUAGE_FEATURE_KIND,
            "query_scope": "language_predicted_task", "temperature": 1.0}
