"""Shared DDP helper utilities for training scripts."""

import os

import torch
import torch.distributed as dist


def is_distributed(cfg):
    """Check whether DDP should be enabled based on config and environment."""
    dist_cfg = cfg.get("training", {}).get("distributed", {})
    if not dist_cfg.get("enabled", False):
        return False
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def init_distributed(cfg):
    """Initialise the DDP process group and return runtime info.

    Returns (rank, world_size, local_rank, device).
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist_cfg = cfg.get("training", {}).get("distributed", {})
    backend = dist_cfg.get("backend", "nccl")
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(
        f"DDP rank {rank}/{world_size} on {device} "
        f"(local_rank={local_rank}, backend={backend})"
    )
    return rank, world_size, local_rank, device


def is_main(rank):
    """True on the rank responsible for logging, checkpointing, and W&B."""
    return rank == 0


def cleanup_distributed():
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
