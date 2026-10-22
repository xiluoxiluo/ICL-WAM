import logging
import json
import inspect
import math
import os
import re
import random
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, GradientAccumulationPlugin
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from .zeva.checkpoint import checkpoint_sha256
from .zeva.ego_pim_schedule import HISTORY_KINDS, apply_history_kind, select_history_kind, eap_lr_factor

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.zeva_training = bool(getattr(self.model, "zeva_enabled", False))
        self.zeva_training_stage = (
            str(getattr(self.model, "zeva_training_stage", "policy_injection"))
            if self.zeva_training
            else None
        )
        if self.zeva_training:
            logger.info("Zeva training stage: %s", self.zeva_training_stage)
        self.zeva_training_recipe = str(getattr(self.model, "zeva_training_recipe", "frozen_parent"))
        self.ego_curriculum = self.zeva_training and self.zeva_training_recipe in {"ego_adapter_warmup", "ego_joint_warmup"}
        self.joint_foundation_training = self.zeva_training and self.zeva_training_recipe == "ego_joint_warmup"
        self.predictive_zeva = self.zeva_training and getattr(
            getattr(getattr(self.model, "zeva_behavior_prefix_adapter", None), "config", None),
            "effect_source", "latest_completed",
        ) == "current_prediction"
        if self.predictive_zeva and self.mixed_precision == "fp16":
            raise ValueError("Predictive Stage2 supports bf16/no precision; fp16 scaler state is not part of its checkpoint contract")
        self._zeva_data_identity = {}
        if self.predictive_zeva:
            for split, dataset in (("train", self.train_dataset), ("validation", self.val_dataset)):
                if dataset is None:
                    raise ValueError("Predictive Stage2 requires separate train and validation prepared datasets")
                self._zeva_data_identity[split] = {
                    "pim_manifest_sha256": checkpoint_sha256(dataset.pim.root / "manifest.json"),
                    "static_context_sha256": checkpoint_sha256(dataset.static_context.path),
                    "num_queries": len(dataset),
                }
        self._zeva_resume_rng = None
        self.ego_warmup_steps = int(cfg.get("zeva_pim_warmup_steps", 500))
        self.ego_available_history_kinds = ()
        if self.ego_curriculum:
            if getattr(self.train_dataset, "history_schedule", None) != "ego_optimizer_step":
                raise ValueError("ego warmup recipes require optimizer-step history candidates")
            self.ego_available_history_kinds = tuple(self.train_dataset.pim.manifest["globally_available_history_kinds"])
            if self.ego_warmup_steps < 1:
                raise ValueError("zeva_pim_warmup_steps must be positive")
        self._validate_policy_warmstart_configuration()

        self.accelerator = Accelerator(
            **({"gradient_accumulation_plugin": GradientAccumulationPlugin(
                num_steps=self.gradient_accumulation_steps, sync_with_dataloader=False,
            )} if self.predictive_zeva else {"gradient_accumulation_steps": self.gradient_accumulation_steps}),
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            # Action-only training does not use the video output head or the
            # final video post-attention branch. Discover those unused weights
            # while keeping a static trainable set across the LR-only warmup.
            kwargs_handlers=([DistributedDataParallelKwargs(
                find_unused_parameters=True, broadcast_buffers=False,
            )] if self.joint_foundation_training else []),
        )
        
        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if self.predictive_zeva and self.accelerator.distributed_type not in {
            DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.MULTI_CPU,
        }:
            raise ValueError("Predictive addon checkpoints support a single process / ordinary DDP, not sharded DeepSpeed/FSDP state")
        deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", {}) or {}
        zero_stage = deepspeed_config.get("zero_optimization", {}).get("stage", "none")
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        self._apply_dit_only_train_mode(self.model)
        if self.zeva_training:
            if self.zeva_training_stage == "policy_injection":
                self._initialize_policy_stage_from_warmstart()
            elif self.zeva_training_stage == "pim_adapter":
                self._initialize_pim_stage_from_policy_checkpoint()
        if self.joint_foundation_training:
            self.model.prepare_zeva_joint_training_precision()
            logger.info("Joint training uses FP32 trainable parameters/Adam states with %s autocast; frozen encoders retain their dtype",
                        self.mixed_precision)
        if self.zeva_training:
            if self.zeva_training_stage == "policy_injection":
                trainable_params = list(self.model.zeva_trainable_parameters())
                optimizer_params = trainable_params
            elif self.zeva_training_stage == "pim_adapter":
                adapter = self.model.zeva_behavior_prefix_adapter
                prompt_params = list(self.model.zeva_prompt_encoder.parameters())
                if self.ego_curriculum:
                    pim_params = prompt_params + list(adapter.pim_adapter_parameters())
                    eap_params = list(adapter.policy_injection_parameters())
                    trainable_params = pim_params + eap_params
                    optimizer_params = [
                        {"params": pim_params, "lr": self.learning_rate, "name": "pim"},
                        {"params": eap_params, "lr": float(cfg.get("zeva_eap_learning_rate", 1e-5)), "name": "eap"},
                    ]
                    if self.joint_foundation_training:
                        foundation_params = self.model.zeva_foundation_parameters()
                        foundation_lr = float(cfg.get("zeva_foundation_learning_rate", 1e-6))
                        if not math.isfinite(foundation_lr) or foundation_lr <= 0:
                            raise ValueError("zeva_foundation_learning_rate must be finite and positive")
                        trainable_params += foundation_params
                        optimizer_params.append({"params": foundation_params, "lr": foundation_lr, "name": "foundation"})
                    logger.info("Predictive PIM: %d PIM-only steps, then %s; effective batch=%d",
                                self.ego_warmup_steps,
                                "foundation+EAP+PIM" if self.joint_foundation_training else "PIM+EAP with frozen FastWAM",
                                self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes)
                    logger.info("Real available history kinds: %s (5:2:1, then 5:2:1:2 with empty)", self.ego_available_history_kinds)
                elif (
                    getattr(adapter.config, "pim_variant", "legacy") == "cross_attempt"
                    or getattr(adapter.config, "prefix_layout", "legacy_merged") == "zeva_ego"
                ):
                    adapter_params = list(adapter.pim_adapter_parameters())
                    trainable_params = prompt_params + adapter_params
                    optimizer_params = [
                        {"params": prompt_params, "lr": self.learning_rate, "name": "pim_readout"},
                        {"params": adapter_params, "lr": self.learning_rate, "name": "pim_injection"},
                    ]
                    logger.info("PIM learning rates: readout=%g injection=%g", self.learning_rate, self.learning_rate)
                else:
                    projector_params = list(adapter.prefix_project.parameters())
                    gate_params = [adapter.pim_gate]
                    trainable_params = prompt_params + projector_params + gate_params
                    optimizer_params = [
                        {"params": prompt_params, "lr": self.learning_rate * 5.0},
                        {"params": projector_params, "lr": self.learning_rate * 5.0},
                        {"params": gate_params, "lr": self.learning_rate},
                    ]
                    logger.info(
                        "PIM LR multipliers: prompt_encoder=5.0 prefix_project=5.0 pim_gate=1.0"
                    )
            else:
                raise RuntimeError(f"unsupported Zeva training stage: {self.zeva_training_stage}")
            if not trainable_params:
                raise ValueError("zeva_enabled model did not expose trainable addon parameters")
            report = self.model.zeva_parameter_report()
            logger.info(
                "Zeva parameter whitelist passed: trainable=%d frozen=%d",
                int(report["trainable_count"]),
                int(report["frozen_count"]),
            )
            trainable_modules = sorted({
                str(name).split(".", 1)[0]
                for name in report["trainable_names"]
            })
            logger.info("Zeva trainable modules: %s", ", ".join(trainable_modules))
        else:
            trainable_params = list(self.model.dit.parameters())
            proprio_encoder = getattr(self.model, "proprio_encoder", None)
            if proprio_encoder is not None:
                trainable_params.extend(list(proprio_encoder.parameters()))
            optimizer_params = trainable_params
        logger.info(
            "trainable parameter count=%d frozen parameter count=%d",
            sum(p.numel() for p in trainable_params),
            sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
        )
        if len({id(parameter) for parameter in trainable_params}) != len(trainable_params):
            raise ValueError("optimizer groups contain duplicate trainable parameters")
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999) if self.joint_foundation_training else (0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        if self.ego_curriculum:
            if str(cfg.lr_scheduler_type) != "constant":
                raise ValueError("ego warmup recipes use constant group LRs after PIM-only warmup")
            if total_train_steps <= self.ego_warmup_steps:
                raise ValueError("formal predictive PIM training must continue past warmup")
            lr_factors = [
                lambda step: 1.0,
                lambda step: eap_lr_factor(step, self.ego_warmup_steps),
            ]
            if self.joint_foundation_training:
                lr_factors.append(lambda step: eap_lr_factor(step, self.ego_warmup_steps))
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_factors)
        else:
            self.scheduler = self._build_scheduler(
                scheduler_type=cfg.lr_scheduler_type,
                total_train_steps=total_train_steps,
                warmup_steps=warmup_steps,
            )
        if self.zeva_training and self.zeva_training_stage == "pim_adapter":
            logger.info(
                "PIM optimizer initial LRs: %s",
                [float(group["lr"]) for group in self.optimizer.param_groups],
            )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        self._zeva_metrics_file = None
        if self.zeva_training and self.accelerator.is_main_process:
            # Joint Stage2 additionally saves a foundation sidecar. Keep the
            # resolved config and scalar trace beside the weight bundle.
            OmegaConf.save(
                config=self.cfg,
                f=os.path.join(self.output_dir, "config.yaml"),
                resolve=True,
            )
            self._zeva_metrics_file = open(
                os.path.join(self.output_dir, "metrics.jsonl"),
                "a",
                encoding="utf-8",
            )

        if bool(getattr(self.model, "compile_training_denoise", False)):
            logger.info("Compiling training denoise forward/backward before DeepSpeed initialization.")
            compile_start = time.perf_counter()
            warmup_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                num_workers=0,
            )
            warmup_sample = next(iter(warmup_loader))
            with self.accelerator.autocast():
                warmup_loss, _ = self.model.training_loss(warmup_sample)
            warmup_loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.accelerator.device)
            logger.info(
                "Finished training denoise compile warmup in %.2f seconds.",
                time.perf_counter() - compile_start,
            )
            set_global_seed(self.seed)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            pad_to_global_batch=getattr(self, "predictive_zeva", False),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            # Predictive inputs/candidate histories are deterministic. Keep
            # worker startup out of the model's flow-noise RNG on resume.
            generator=(torch.Generator().manual_seed(self.seed + self.accelerator.process_index)
                       if getattr(self, "predictive_zeva", False) else None),
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            def cosine_factor(step: int) -> float:
                progress = min(
                    max(float(step) / float(max(remaining_steps, 1)), 0.0),
                    1.0,
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Apply one scalar to every group so their initial LR ratios
                # remain unchanged throughout warmup and cosine decay.
                return 0.01 + 0.99 * cosine

            main_scheduler = LambdaLR(self.optimizer, lr_lambda=cosine_factor)
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _validate_policy_warmstart_configuration(self) -> None:
        zeva_cfg = self.cfg.model.get("zeva", {})
        value = zeva_cfg.get("policy_warmstart_checkpoint") if hasattr(zeva_cfg, "get") else None
        if value in (None, "", "None", "null"):
            return
        if not self.zeva_training or self.zeva_training_stage != "policy_injection":
            raise ValueError("policy_warmstart_checkpoint is only valid for Stage2A policy_injection training")
        if self.resume not in (None, "", "None", "null"):
            raise ValueError("policy_warmstart_checkpoint cannot be combined with resume; clear it when resuming")
        if zeva_cfg.get("policy_checkpoint") not in (None, "", "None", "null"):
            raise ValueError("policy_warmstart_checkpoint and policy_checkpoint are mutually exclusive")
        layout = getattr(getattr(self.model.zeva_behavior_prefix_adapter, "config", None), "prefix_layout", "legacy_merged")
        if layout != "zeva_ego":
            raise ValueError("policy_warmstart_checkpoint requires the new zeva_ego prefix_layout")

    def _initialize_policy_stage_from_warmstart(self) -> None:
        """Reuse old Stage2A weights while training the new effect projector.

        This is an explicit one-time initialization, never a resume or a way
        to import an untrained effect projector into frozen-parent Stage2B.
        """
        self._validate_policy_warmstart_configuration()
        zeva_cfg = self.cfg.model.get("zeva", {})
        value = zeva_cfg.get("policy_warmstart_checkpoint")
        if value in (None, "", "None", "null"):
            return
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"zeva.policy_warmstart_checkpoint does not exist: {path}")
        base_path = self.cfg.get("ckpt")
        cte_path = zeva_cfg.get("cte", {}).get("checkpoint")
        if base_path in (None, "", "None", "null"):
            raise ValueError("policy_warmstart initialization requires cfg.ckpt")
        if cte_path in (None, "", "None", "null"):
            raise ValueError("policy_warmstart initialization requires model.zeva.cte.checkpoint")
        self.model.load_zeva_addon_checkpoint(
            str(path),
            base_checkpoint_sha256=checkpoint_sha256(str(base_path)),
            cte_checkpoint_sha256=checkpoint_sha256(str(cte_path)),
            load_scope="policy_warmstart",
        )
        self.model.zeva_behavior_prefix_adapter.reset_pim_parameters()
        self.model.configure_zeva_trainable_state()
        logger.info(
            "Warmstarted zeva_ego Stage2A from %s (sha256=%s); effect projector remains trainable",
            path, checkpoint_sha256(path),
        )

    def _get_zeva_policy_checkpoint(self) -> Path | None:
        if not self.zeva_training:
            return None
        zeva_cfg = self.cfg.model.get("zeva", {})
        value = zeva_cfg.get("policy_checkpoint") if hasattr(zeva_cfg, "get") else None
        if value in (None, "", "None", "null"):
            return None
        path = Path(str(value))
        if not path.is_file():
            raise FileNotFoundError(f"zeva.policy_checkpoint does not exist: {path}")
        return path

    def _initialize_pim_stage_from_policy_checkpoint(self) -> None:
        # A resume always restores the current PIM stage and must not be
        # overwritten by the one-time Stage-2A initialization path.  Keep
        # this guard before looking up policy_checkpoint: that path may point
        # to an unavailable location from the machine that created the run.
        if self.resume not in (None, "", "None", "null"):
            return
        policy_checkpoint = self._get_zeva_policy_checkpoint()
        if policy_checkpoint is None:
            raise ValueError(
                "pim_adapter training requires model.zeva.policy_checkpoint when resume is not set"
            )
        base_path = self.cfg.get("ckpt")
        zeva_cfg = self.cfg.model.get("zeva", {})
        cte_path = zeva_cfg.get("cte", {}).get("checkpoint") if hasattr(zeva_cfg, "get") else None
        if base_path in (None, "", "None", "null"):
            raise ValueError("pim_adapter initialization requires cfg.ckpt")
        if cte_path in (None, "", "None", "null"):
            raise ValueError("pim_adapter initialization requires model.zeva.cte.checkpoint")
        model = self.model
        payload = model.load_zeva_addon_checkpoint(
            str(policy_checkpoint),
            base_checkpoint_sha256=checkpoint_sha256(str(base_path)),
            cte_checkpoint_sha256=checkpoint_sha256(str(cte_path)),
            load_scope="policy_injection",
        )
        if payload.get("training_stage") != "policy_injection":
            raise ValueError("model.zeva.policy_checkpoint must be a policy_injection checkpoint")
        model.zeva_behavior_prefix_adapter.reset_pim_parameters()
        model.configure_zeva_trainable_state()
        logger.info("Initialized pim_adapter from policy checkpoint: %s", policy_checkpoint)

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            if getattr(self, "predictive_zeva", False):
                required = ("optimizer_scheduler.pt", "trainer_state.json", "COMPLETE",
                            f"rng_rank_{self.accelerator.process_index:04d}.pt")
                missing = [name for name in required if not (resume_path / name).is_file()]
                if missing:
                    raise ValueError(f"Predictive Stage2 resume requires a complete checkpoint directory; missing {missing}")
            if self.zeva_training and (resume_path / "optimizer_scheduler.pt").is_file():
                logger.info("Resuming Zeva addon optimizer/scheduler state from directory: %s", resume)
                addon_path = (
                    resume_path.parent.parent
                    / "weights"
                    / f"{resume_path.name}_addon.pt"
                )
                if not addon_path.is_file():
                    raise FileNotFoundError(
                        f"Zeva resume state is missing its addon weights: {addon_path}"
                    )
                base_path = self.cfg.get("ckpt")
                zeva_cfg = self.cfg.model.get("zeva", {})
                cte_path = zeva_cfg.get("cte", {}).get("checkpoint") if hasattr(zeva_cfg, "get") else None
                if base_path in (None, "", "None", "null") or cte_path in (None, "", "None", "null"):
                    raise ValueError("Zeva resume requires cfg.ckpt and zeva.cte.checkpoint")
                model = self.accelerator.unwrap_model(self.model)
                addon_payload = model.load_zeva_addon_checkpoint(
                    str(addon_path),
                    base_checkpoint_sha256=checkpoint_sha256(str(base_path)),
                    cte_checkpoint_sha256=checkpoint_sha256(str(cte_path)),
                    load_scope="all",
                )
                checkpoint_stage = addon_payload.get("training_stage")
                self._validate_zeva_resume_recipe(addon_payload)
                if checkpoint_stage != self.zeva_training_stage:
                    raise ValueError(
                        "Zeva resume stage mismatch: "
                        f"checkpoint={checkpoint_stage}, current={self.zeva_training_stage}"
                    )
                self.addon_weights_path = str(addon_path)
                payload = torch.load(resume_path / "optimizer_scheduler.pt", map_location="cpu")
                self.optimizer.load_state_dict(payload["optimizer"])
                self.scheduler.load_state_dict(payload["scheduler"])
                state_file = resume_path / "trainer_state.json"
                if state_file.is_file():
                    with state_file.open("r", encoding="utf-8") as handle:
                        progress = json.load(handle)
                    self.global_step = int(progress.get("global_step", 0))
                    self.epoch = int(progress.get("epoch", 0))
                    self.batch_in_epoch = int(progress.get("batch_in_epoch", 0))
                if getattr(self, "predictive_zeva", False):
                    if progress.get("data_identity") != self._zeva_data_identity:
                        raise ValueError("Predictive Stage2 resume train/validation cache identity changed")
                    if int(addon_payload.get("step", -1)) != self.global_step:
                        raise ValueError("Predictive Stage2 resume addon/progress step mismatch")
                    self._zeva_resume_rng = torch.load(
                        resume_path / f"rng_rank_{self.accelerator.process_index:04d}.pt",
                        map_location="cpu", weights_only=False,
                    )
                    self._validate_rank_rng(self._zeva_resume_rng)
                    self.train_sampler.set_epoch_offset(self.epoch)
                    self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                    if self.batch_in_epoch > 0 and len(self.train_sampler) == 0:
                        # DataLoaderShard does not advance its iteration for
                        # an initially empty loader. Normalize an exhausted
                        # checkpoint to the next epoch before constructing it.
                        self.epoch += 1
                        self.batch_in_epoch = 0
                        self.train_sampler.set_epoch_offset(self.epoch)
                        self.train_sampler.clear_resume_batch_offset()
                return
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if self.zeva_training:
            if getattr(self, "predictive_zeva", False):
                raise ValueError("Predictive Stage2 resume requires checkpoints/state/step_XXXXXX with optimizer and rank RNG; addon-only is not a resume")
            base_path = self.cfg.get("ckpt")
            zeva_cfg = self.cfg.model.get("zeva", {})
            cte_path = zeva_cfg.get("cte", {}).get("checkpoint") if hasattr(zeva_cfg, "get") else None
            if base_path in (None, "", "None", "null") or cte_path in (None, "", "None", "null"):
                raise ValueError("Zeva resume requires cfg.ckpt and zeva.cte.checkpoint")
            model = self.accelerator.unwrap_model(self.model)
            payload = model.load_zeva_addon_checkpoint(
                str(resume_path),
                base_checkpoint_sha256=checkpoint_sha256(str(base_path)),
                cte_checkpoint_sha256=checkpoint_sha256(str(cte_path)),
                load_scope="all",
            )
            checkpoint_stage = payload.get("training_stage")
            self._validate_zeva_resume_recipe(payload)
            if checkpoint_stage != self.zeva_training_stage:
                raise ValueError(
                    "Zeva resume stage mismatch: "
                    f"checkpoint={checkpoint_stage}, current={self.zeva_training_stage}"
                )
            self.addon_weights_path = str(resume_path)
            logger.info("Loaded Zeva addon weights only: %s", resume)
            return
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _validate_zeva_resume_recipe(self, payload):
        expected = getattr(self, "zeva_training_recipe", "frozen_parent")
        if payload.get("training_recipe", "frozen_parent") != expected:
            raise ValueError("Zeva resume training_recipe mismatch")
        keys = set()
        if getattr(self, "predictive_zeva", False):
            keys.update(("learning_rate", "weight_decay", "max_grad_norm", "lr_scheduler_type",
                         "max_steps", "num_epochs", "seed", "batch_size",
                         "gradient_accumulation_steps", "mixed_precision"))
        if expected in {"ego_adapter_warmup", "ego_joint_warmup"}:
            keys.update(("zeva_pim_warmup_steps", "zeva_eap_learning_rate", "learning_rate", "seed",
                         "gradient_accumulation_steps", "batch_size"))
        if expected == "ego_joint_warmup":
            keys.add("zeva_foundation_learning_rate")
        saved = payload.get("config", {})
        for key in sorted(keys):
            if saved.get(key) != self.cfg.get(key):
                raise ValueError(f"Predictive Stage2 resume curriculum/optimizer mismatch: {key}")

    def _rank_rng_state(self):
        device = torch.device(self.accelerator.device)
        return {
            "format": "zeva_ego_stage2_rng_v1", "step": self.global_step,
            "world_size": self.accelerator.num_processes, "rank": self.accelerator.process_index,
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }

    def _validate_rank_rng(self, state):
        for name, expected in (("format", "zeva_ego_stage2_rng_v1"), ("step", self.global_step),
                               ("world_size", self.accelerator.num_processes), ("rank", self.accelerator.process_index)):
            if state.get(name) != expected:
                raise ValueError(f"Predictive Stage2 rank RNG mismatch: {name}")
        if (state.get("torch_cuda") is not None) != (torch.device(self.accelerator.device).type == "cuda"):
            raise ValueError("Predictive Stage2 resume RNG device type changed")

    def _restore_rank_rng(self, state):
        self._validate_rank_rng(state)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if state["torch_cuda"] is not None:
            torch.cuda.set_rng_state(state["torch_cuda"].cpu(), self.accelerator.device)

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        if bool(getattr(model, "zeva_enabled", False)):
            model.configure_zeva_trainable_state()
            return
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def _evaluate_zeva(self):
        """Held-out action loss with paired PIM-on/off random draws.

        Prepared Zeva samples contain one image plus causal memory. They must
        not enter the generic video rollout evaluator, which drops that state.
        This is an offline validation metric, not RoboTwin success rate.
        """
        model = self.accelerator.unwrap_model(self.model)
        per_rank = int(self.cfg.get("eval_num_samples", 16))
        dataset_size = len(self.val_dataset)
        if per_rank < 1 or dataset_size < 1:
            raise ValueError("Zeva validation requires a nonempty dataset and eval_num_samples >= 1")
        world_size = self.accelerator.num_processes
        total_samples = min(dataset_size, per_rank * world_size)
        count = (total_samples + world_size - 1) // world_size
        module_modes = {module: module.training for module in model.modules()}
        model.eval()
        device = torch.device(self.accelerator.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        rows = []
        try:
            for offset in range(count):
                position = self.accelerator.process_index + offset * world_size
                valid = position < total_samples
                # Spread the fixed validation budget across the entire cache,
                # whose rows are grouped by episode. Rank-striding only the
                # first rows repeatedly evaluates one episode. Pad the final
                # distributed batch but exclude padding from the metric mean.
                position = min(position, total_samples - 1)
                index = (
                    position * (dataset_size - 1) // (total_samples - 1)
                    if total_samples > 1 else dataset_size // 2
                )
                # Keep validation out of the training RNG stream and reuse
                # precisely the same noise/timesteps for the paired call. The
                # dataset read is included in case transforms consume RNG.
                with torch.random.fork_rng(devices=cuda_devices):
                    source = self.val_dataset[index]
                    unbatched = source["video"].ndim == 4
                    if source["video"].ndim not in (4, 5):
                        raise ValueError("Zeva validation video must be [3,T,H,W] or [B,3,T,H,W]")
                    sample = {
                        key: value.unsqueeze(0) if unbatched and isinstance(value, torch.Tensor) else value
                        for key, value in source.items()
                    }
                    sample["_training_mode"] = "zeva_stage2"
                    if getattr(self, "ego_curriculum", False):
                        # Validation uses a fixed real nonempty mode so its
                        # paired on/off gain remains interpretable after step500.
                        kind = HISTORY_KINDS.index(self.ego_available_history_kinds[0])
                        sample = apply_history_kind(sample, kind)
                    memory_off = dict(sample)
                    for key in ("pim_phases", "pim_effects", "pim_mask"):
                        if key in memory_off:
                            memory_off[key] = torch.zeros_like(memory_off[key])
                    cpu_state = torch.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state(device) if cuda_devices else None
                    with self.accelerator.autocast():
                        on_loss, on_metrics = model(sample)
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state, device)
                    with self.accelerator.autocast():
                        off_loss, _ = model(memory_off)
                values = [
                    float(on_loss), float(off_loss), float(off_loss - on_loss),
                    float(on_metrics.get("loss_flow", on_loss)),
                    float(on_metrics.get("behavior_prior_nll", 0.0)),
                ]
                rows.append([value if valid else 0.0 for value in values] + [float(valid)])
            gathered = self.accelerator.gather_for_metrics(torch.tensor(rows, device=device, dtype=torch.float32))
            mean = (gathered[:, :5].sum(dim=0) / gathered[:, 5].sum()).cpu().tolist()
            return dict(zip(
                ("val_loss", "val_loss_memory_off", "val_memory_gain", "val_loss_flow", "val_prior_nll"), mean,
            ))
        finally:
            # Stage2B intentionally has model.training=False while only the
            # addons train. Restore every module, not just model.dit.training.
            for module, training in module_modes.items():
                module.training = training

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        if self.zeva_training:
            return self._evaluate_zeva()

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        if self.zeva_training:
            addon_path = os.path.join(self.weights_dir, f"{step_tag}_addon.pt")
            zeva_cfg = self.cfg.model.get("zeva", {})
            cte_path = zeva_cfg.get("cte", {}).get("checkpoint") if hasattr(zeva_cfg, "get") else None
            base_path = self.cfg.get("ckpt")
            if base_path in (None, "", "None", "null") or not Path(str(base_path)).is_file():
                raise FileNotFoundError("Zeva Stage 2 requires cfg.ckpt to be an existing base checkpoint")
            if cte_path in (None, "", "None", "null") or not Path(str(cte_path)).is_file():
                raise FileNotFoundError("Zeva Stage 2 requires zeva.cte.checkpoint to be an existing CTE checkpoint")
            base_hash = checkpoint_sha256(str(base_path))
            cte_hash = checkpoint_sha256(str(cte_path))
            for artifact_dir in (Path(self.weights_dir), Path(self.output_dir)):
                artifact_dir.joinpath("base_checkpoint.sha256").write_text(
                    base_hash + "\n", encoding="utf-8"
                )
                artifact_dir.joinpath("cte_checkpoint.sha256").write_text(
                    cte_hash + "\n", encoding="utf-8"
                )
            cache_path = zeva_cfg.get("cache", {}).get("path") if hasattr(zeva_cfg, "get") else None
            if cache_path not in (None, "", "None", "null"):
                manifest = Path(str(cache_path)) / "manifest.json"
                if manifest.is_file():
                    manifest_text = manifest.read_text(encoding="utf-8")
                    for artifact_dir in (Path(self.weights_dir), Path(self.output_dir)):
                        artifact_dir.joinpath("cache_manifest.json").write_text(
                            manifest_text, encoding="utf-8"
                        )
            model.save_zeva_addon_checkpoint(
                addon_path,
                step=self.global_step,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                base_checkpoint_sha256=base_hash,
                cte_checkpoint_sha256=cte_hash,
            )
            self.addon_weights_path = addon_path
            return addon_path
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        if getattr(self, "predictive_zeva", False):
            payload["data_identity"] = self._zeva_data_identity
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        state_path = os.path.join(self.state_dir, step_tag)

        self.accelerator.wait_for_everyone()
        ensure_dir(state_path)
        if getattr(self, "predictive_zeva", False) and self.accelerator.is_main_process:
            # An interrupted overwrite must never retain a stale completion
            # marker beside a mix of old/new weights, optimizer and RNG state.
            (Path(state_path) / "COMPLETE").unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        if self.zeva_training:
            # Weights were saved above: addon-only for frozen recipes, or an
            # addon plus the complete policy-backbone sidecar for joint mode.
            # VAE/text/CTE stay frozen and are not duplicated here.
            if self.accelerator.is_main_process:
                torch.save(
                    {
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                    },
                    Path(state_path) / "optimizer_scheduler.pt",
                )
            if getattr(self, "predictive_zeva", False):
                rng_path = Path(state_path) / f"rng_rank_{self.accelerator.process_index:04d}.pt"
                temporary = rng_path.with_suffix(".tmp")
                torch.save(self._rank_rng_state(), temporary)
                temporary.replace(rng_path)
        else:
            self.accelerator.save_state(output_dir=state_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
            if getattr(self, "predictive_zeva", False):
                (Path(state_path) / "COMPLETE").write_text("complete\n", encoding="utf-8")
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        if getattr(self, "_zeva_resume_rng", None) is not None:
            self._restore_rank_rng(self._zeva_resume_rng)
            self._zeva_resume_rng = None
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    if self.zeva_training:
                        # Route through the prepared wrapper so DDP/
                        # DeepSpeed forward hooks observe the complete Zeva
                        # prompt -> adapter -> action-loss graph.
                        sample = dict(sample)
                        if self.ego_curriculum:
                            kind = select_history_kind(
                                self.global_step, seed=self.seed, warmup_steps=self.ego_warmup_steps,
                                available_kinds=self.ego_available_history_kinds,
                            )
                            sample = apply_history_kind(sample, kind)
                        sample["_training_mode"] = "zeva_stage2"
                        loss, loss_dict = self.model(sample)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"Non-finite Zeva Stage 2 action loss at global step {self.global_step + 1}: {loss.item()}"
                            )
                    else:
                        loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    clip_params = (parameter for group in self.optimizer.param_groups if group["lr"] > 0
                                   for parameter in group["params"]) if self.ego_curriculum else self.model.parameters()
                    grad_norm = self.accelerator.clip_grad_norm_(clip_params, self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ego_curriculum and self.accelerator.optimizer_step_was_skipped:
                        # Overflow is not an optimizer update and must not
                        # advance the warmup or history-mode clock.
                        continue
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.zeva_training
                        and self.accelerator.is_main_process
                        and self._zeva_metrics_file is not None
                    ):
                        self._zeva_metrics_file.write(
                            json.dumps(
                                {
                                    "step": int(self.global_step),
                                    "loss": float(global_loss),
                                    "grad_norm": float(global_grad_norm),
                                    "lr": float(current_lr),
                                    **{
                                        str(key): float(value)
                                        for key, value in global_loss_metrics.items()
                                    },
                                }
                            )
                            + "\n"
                        )
                        self._zeva_metrics_file.flush()

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "val_loss_memory_off" in metrics:
                                description += " memory_off_loss=%.4f memory_gain=%.4f" % (
                                    metrics["val_loss_memory_off"], metrics["val_memory_gain"],
                                )
                            if "psnr_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (metrics["psnr_rd"], metrics["ssim_rd"])
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {f"eval/{key}": float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
                            self._wandb_log(eval_payload)

                    if (self.save_every > 0 and self.global_step % self.save_every == 0
                            and self.global_step < self.max_steps):
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        if self._zeva_metrics_file is not None:
                            self._zeva_metrics_file.close()
                            self._zeva_metrics_file = None
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        if self._zeva_metrics_file is not None:
            self._zeva_metrics_file.close()
            self._zeva_metrics_file = None
        
