"""
Single-process pretraining without CUDA assumptions.

Uses the same Hydra config as ``pretrain.py`` (``cfg_pretrain``). Set ``URM_DEVICE`` to choose the
device (default ``cpu``). Optional: ``URM_COMPILE_CPU=1`` to enable ``torch.compile`` on CPU/MPS
(first iterations can be very slow).

Distributed / multi-GPU training is not supported here; use ``pretrain.py`` with CUDA.
"""

from __future__ import annotations

import copy
import os
from typing import Any, List, Optional

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import wandb

try:
    from adam_atan2 import AdamATan2
except ImportError:
    from adam_atan2_pytorch import AdamAtan2 as AdamATan2

from models.muon import Muon
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from pretrain import (
    EMAHelper,
    PretrainConfig,
    TrainState,
    _get_loop_config,
    _resize_puzzle_embedding_if_needed,
    _resolve_checkpoint_path,
    compute_lr,
    create_evaluators,
    load_synced_config,
    save_code_and_config,
)
from utils import load_model_class


def train_device() -> torch.device:
    return torch.device(os.environ.get("URM_DEVICE", "cpu"))


def create_dataloader(
    config: PretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    device: torch.device,
    **kwargs,
):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    pin = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=pin,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    device: torch.device,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model_config = None
    with torch.device(device):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        model_config = getattr(getattr(model, "model", None), "config", None)

    should_compile = False
    if device.type == "cuda":
        should_compile = (
            "DISABLE_COMPILE" not in os.environ
            and (model_config is None or not getattr(model_config, "profile", False))
        )
    elif device.type in ("cpu", "mps") and os.environ.get("URM_COMPILE_CPU") == "1":
        should_compile = model_config is None or not getattr(model_config, "profile", False)

    if should_compile:
        model = torch.compile(model, dynamic=False)  # type: ignore

    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    if config.use_muon:
        adam_params = [p for p in model.parameters() if p.ndim != 2]
        muon_params = [p for p in model.parameters() if p.ndim == 2]

        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.sparse_optimizer_tensors(),  # type: ignore
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            Muon(
                [
                    {
                        "params": muon_params,
                        "use_muon": True,
                        "lr": 1e-4,
                    },
                    {
                        "params": adam_params,
                        "use_muon": False,
                        "lr": 1e-4,
                        "weight_decay": 0.1,
                        "adamw_betas": (0.9, 0.95),
                        "adamw_eps": 1e-8,
                    },
                ]
            ),
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.sparse_optimizer_tensors(),  # type: ignore
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            AdamATan2(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            ),
        ]

    optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    device: torch.device,
):
    effective_gbs = config.global_batch_size * max(1, config.grad_accum_steps)
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / effective_gbs
    )

    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata, rank=rank, world_size=world_size, device=device
    )

    train_state = TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )

    load_checkpoint(train_state, config, rank, device)
    return train_state


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    state = {
        "step": train_state.step,
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_states": [optim.state_dict() for optim in train_state.optimizers],
    }

    state["rng_state"] = torch.random.get_rng_state()
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

    torch.save(state, os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt"))


def load_checkpoint(train_state: TrainState, config: PretrainConfig, rank: int, device: torch.device):
    load_path = config.load_checkpoint
    if load_path is None:
        return

    if load_path == "latest":
        if config.checkpoint_path is None:
            raise ValueError("Cannot load latest checkpoint without a checkpoint_path configured.")
        load_path = config.checkpoint_path

    resolved_path = _resolve_checkpoint_path(load_path)
    if resolved_path is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from '{load_path}'")

    if rank == 0:
        print(f"Loading checkpoint {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location=device)

    def _prepare_rng_state(state: Any, dev: str | None) -> Any:
        if state is None:
            return None
        if isinstance(state, (list, tuple)):
            return [_prepare_rng_state(s, dev) for s in state]
        tensor_state = torch.as_tensor(state, device=dev)
        if tensor_state.dtype != torch.uint8:
            tensor_state = tensor_state.to(torch.uint8)
        return tensor_state

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        optimizer_states = checkpoint.get("optimizer_states")
        step = checkpoint.get("step")
        rng_state = checkpoint.get("rng_state")
        cuda_rng_state = checkpoint.get("cuda_rng_state")
    else:
        state_dict = checkpoint
        optimizer_states = None
        step = None
        rng_state = None
        cuda_rng_state = None

    _resize_puzzle_embedding_if_needed(train_state.model, state_dict)
    load_result = train_state.model.load_state_dict(state_dict, strict=config.load_strict, assign=True)

    if not config.load_strict and rank == 0:
        missing, unexpected = load_result
        if missing:
            print(f"Warning: missing keys during checkpoint load: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys during checkpoint load: {unexpected}")

    if optimizer_states is not None:
        if not config.load_optimizer_state:
            if rank == 0:
                print("Skipping optimizer state load because load_optimizer_state=False")
        elif len(optimizer_states) != len(train_state.optimizers):
            raise ValueError(
                "Checkpoint optimizer count does not match current configuration: "
                f"{len(optimizer_states)} vs {len(train_state.optimizers)}"
            )
        else:
            for optimizer, optimizer_state in zip(train_state.optimizers, optimizer_states):
                optimizer.load_state_dict(optimizer_state)

    if step is not None:
        train_state.step = int(step)

    train_state.carry = None

    if rng_state is not None:
        normalized_rng_state = _prepare_rng_state(rng_state, device="cpu")
        if isinstance(normalized_rng_state, list):
            normalized_rng_state = normalized_rng_state[0]
        torch.random.set_rng_state(normalized_rng_state)

    if cuda_rng_state is not None and torch.cuda.is_available():
        normalized_cuda_state = _prepare_rng_state(cuda_rng_state, device="cpu")
        try:
            if isinstance(normalized_cuda_state, list):
                if len(normalized_cuda_state) != torch.cuda.device_count():
                    primary_state = normalized_cuda_state[0]
                    normalized_cuda_state = [primary_state for _ in range(torch.cuda.device_count())]
                torch.cuda.set_rng_state_all(normalized_cuda_state)
            else:
                torch.cuda.set_rng_state(normalized_cuda_state)
        except RuntimeError:
            fallback_state = (
                normalized_cuda_state[0]
                if isinstance(normalized_cuda_state, list)
                else normalized_cuda_state
            )
            torch.cuda.set_rng_state(fallback_state)


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    accum_steps = max(1, getattr(config, "grad_accum_steps", 1))
    if train_state.step >= train_state.total_steps:
        return

    batch = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device(device):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    compute_target_q = train_state.step % config.target_q_update_every == 0
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[], compute_target_q=compute_target_q
    )

    loss_scale = 1.0 / (global_batch_size * accum_steps)
    (loss_scale * loss).backward()
    train_state.accum_step += 1

    should_step = train_state.accum_step % accum_steps == 0
    if not should_step:
        return

    if world_size > 1:
        for param in train_state.model.parameters():
            if not param.requires_grad:
                continue
            grad = param.grad
            if grad is None:
                grad = torch.zeros_like(param)
            dist.all_reduce(grad)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    train_state.step += 1
    train_state.accum_step = 0

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics.get("count", 0), 1)

            def _normalize_metric(key: str, value: float) -> float:
                if key.startswith("profile/"):
                    return value / world_size
                if key.endswith("loss"):
                    return value / global_batch_size
                return value / count

            reduced_metrics = {f"train/{k}": _normalize_metric(k, v) for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    device: torch.device,
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds: dict = {}
        metric_keys: List[str] = []
        metric_values = None
        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            batch = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}
            with torch.device(device):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device=device,
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics

        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        if config.checkpoint_path is not None and len(save_preds):
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds,
                os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"),
            )

        del save_preds

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")

            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            metrics = evaluator.result(
                evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group
            )
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")

        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics


def _hydra_save_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        return str(HydraConfig.get().runtime.output_dir)
    except Exception:
        return os.getcwd()


@hydra.main(config_path="config", config_name="cfg_pretrain_cpu", version_base=None)
def launch(hydra_config: DictConfig):
    if "LOCAL_RANK" in os.environ:
        raise RuntimeError(
            "pretrain_cpu.py does not support torchrun / LOCAL_RANK. "
            "Use pretrain.py with CUDA for distributed training."
        )

    device = train_device()
    print(f"pretrain_cpu: using device {device} (set URM_DEVICE to override)")

    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)

    train_epochs_per_iter = config.eval_interval
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
        device=device,
    )

    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
            device=device,
        )
        evaluators = create_evaluators(config, eval_metadata)
    except FileNotFoundError:
        print("eval metadata FileNotFoundError")
        eval_loader = eval_metadata = None
        evaluators = []

    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE, device=device)

    ema_helper = None
    if config.ema:
        print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if train_state.step > 0:
            progress_bar.update(train_state.step)

        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config, _hydra_save_dir())

    for _iter_id in range(total_iters):
        if RANK == 0:
            count = 0
            for set_name, batch, global_batch_size in train_loader:
                count += 1
            print(f"_iter_id: {_iter_id}")
            print(f"train_epochs_per_iter: {train_epochs_per_iter}")
            print(f"total_iters: {total_iters}")
            print(f"train_loader len: {count}")
            print(f"Epoch {_iter_id * train_epochs_per_iter}")

        train_state.model.train()

        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
                device=device,
            )

            if config.ema and ema_helper is not None:
                ema_helper.update(train_state.model)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        if eval_loader is not None and eval_metadata is not None:
            if config.ema and ema_helper is not None:
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            loop_config = _get_loop_config(train_state_eval.model)
            if loop_config is not None:
                original_loops = loop_config.loops
                if len(config.loop_deltas) == 0:
                    config.loop_deltas = [0, 8]
                else:
                    config.loop_deltas = [0]
            for delta in config.loop_deltas:
                if loop_config is not None:
                    loop_config.loops = original_loops + delta

                metrics = evaluate(
                    config,
                    train_state_eval,
                    eval_loader,
                    eval_metadata,
                    evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                    device=device,
                )
                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)

            if loop_config is not None:
                loop_config.loops = original_loops

            if config.ema and ema_helper is not None and train_state_eval is not train_state:
                del train_state_eval

        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            if config.ema and ema_helper is not None:
                ts_to_save = copy.deepcopy(train_state)
                ts_to_save.model = ema_helper.ema_copy(ts_to_save.model)
                save_train_state(config, ts_to_save)
                del ts_to_save
            else:
                save_train_state(config, train_state)

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
