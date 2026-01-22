import os
import torch

if not os.getcwd().endswith("root-obj-perf"):
    os.chdir("root-obj-perf")


def extract_wandb_run_id(run_path):
    return run_path[run_path.rfind("/") + 1 :]


def get_wandb_save_path(run_id, wandb_dir=None):
    wandb_dir = os.path.join(os.getcwd(), "wandb") if wandb_dir is None else wandb_dir
    run_dirs = os.listdir(os.path.join(os.getcwd(), "wandb"))
    run_dir_needed = ""
    for run_dir in run_dirs:
        if run_dir.endswith(run_id):
            run_dir_needed = run_dir
            break
    run_dir_path = os.path.join(wandb_dir, run_dir_needed)
    return run_dir_path


def get_model_ckpt(run_id, ckpt_name="best_part_model.pth", wandb_dir="wandb"):
    run_dir_path = get_wandb_save_path(run_id, wandb_dir=wandb_dir)
    ckpt_path = os.path.join(run_dir_path, "files", ckpt_name)
    ckpt = torch.load(ckpt_path, weights_only=True)
    return ckpt
