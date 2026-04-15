"""Reproducibility helpers for training and evaluation."""

import json
import os
import random

import numpy as np
import torch


class EpochShuffleTensorPool:
    """Serve fixed tensors with a deterministic shuffle per epoch."""

    def __init__(self, values, base_seed):
        values = torch.as_tensor(values, dtype=torch.float32).flatten().cpu()
        if values.numel() == 0:
            raise ValueError("values must not be empty")
        self.values = values
        self.base_seed = int(base_seed)
        self.current_epoch = None
        self.current_order = values
        self.position = 0

    def set_epoch(self, epoch):
        """Shuffle the pool deterministically for the given epoch and reset the cursor."""
        epoch = int(epoch)
        generator = _cpu_generator(self.base_seed + epoch)
        permutation = torch.randperm(self.values.numel(), generator=generator)
        self.current_order = self.values[permutation]
        self.current_epoch = epoch
        self.position = 0

    def next_batch(self, batch_size, device=None, dtype=torch.float32):
        """Return the next batch from the current epoch order and wrap if needed."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.current_epoch is None:
            self.set_epoch(0)

        chunks = []
        remaining = batch_size
        while remaining > 0:
            available = self.current_order.numel() - self.position
            take = min(remaining, available)
            chunks.append(self.current_order[self.position : self.position + take])
            self.position = (self.position + take) % self.current_order.numel()
            remaining -= take

        batch = torch.cat(chunks, dim=0)
        if device is not None:
            batch = batch.to(device=device, dtype=dtype)
        else:
            batch = batch.to(dtype=dtype)
        return batch


def set_global_seed(seed):
    """Seed Python, NumPy, and Torch RNGs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _cpu_generator(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _resolve_center_range(params, attribute_name):
    center_range = getattr(params, attribute_name, None)
    if center_range is not None and len(center_range) == 2:
        center_min, center_max = float(center_range[0]), float(center_range[1])
        if center_min > center_max:
            center_min, center_max = center_max, center_min
        return center_min, center_max

    wave_min = float(params.wavelength_range[0])
    wave_max = float(params.wavelength_range[1])
    width = float(getattr(params, "lorentz_width", 0.05))
    padding = max((wave_max - wave_min) * 0.1, width * 2)
    return wave_min + padding, wave_max - padding


def _save_center_pool(centers, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("index,center_um\n")
        for index, value in enumerate(centers.tolist(), start=1):
            file.write(f"{index},{value:.8f}\n")


def prepare_reproducibility_assets(params, run_dir):
    """Create fixed pools used for deterministic training and evaluation."""
    reproducibility_dir = os.path.join(run_dir, "reproducibility")
    os.makedirs(reproducibility_dir, exist_ok=True)

    assets = {"reproducibility_dir": reproducibility_dir}
    metadata = {
        "seed": int(getattr(params, "seed", 0)),
        "fix_training_target_centers": bool(getattr(params, "fix_training_target_centers", False)),
        "center_pool_size": int(getattr(params, "center_pool_size", 0)),
        "shuffle_training_target_centers_each_epoch": bool(getattr(params, "fix_training_target_centers", False)),
        "fix_q_evaluation_noise": bool(getattr(params, "fix_q_evaluation_noise", False)),
        "q_eval_num_samples": int(getattr(params, "q_eval_num_samples", 0)),
    }

    if getattr(params, "fix_training_target_centers", False):
        center_min_1, center_max_1 = _resolve_center_range(params, "lorentz_center_range_1")
        center_min_2, center_max_2 = _resolve_center_range(params, "lorentz_center_range_2")
        center_pool_size = max(1, int(getattr(params, "center_pool_size", 0)))
        center_generator_1 = _cpu_generator(int(getattr(params, "seed", 0)) + 101)
        center_generator_2 = _cpu_generator(int(getattr(params, "seed", 0)) + 102)
        center_pool_1 = torch.empty(center_pool_size, dtype=torch.float32)
        center_pool_2 = torch.empty(center_pool_size, dtype=torch.float32)
        center_pool_1.uniform_(center_min_1, center_max_1, generator=center_generator_1)
        center_pool_2.uniform_(center_min_2, center_max_2, generator=center_generator_2)

        center_pool_path_1 = os.path.join(reproducibility_dir, "training_target_center_pool_1.csv")
        center_pool_path_2 = os.path.join(reproducibility_dir, "training_target_center_pool_2.csv")
        _save_center_pool(center_pool_1, center_pool_path_1)
        _save_center_pool(center_pool_2, center_pool_path_2)

        assets["training_target_center_pool_1"] = EpochShuffleTensorPool(
            center_pool_1,
            base_seed=int(getattr(params, "seed", 0)) + 401,
        )
        assets["training_target_center_pool_2"] = EpochShuffleTensorPool(
            center_pool_2,
            base_seed=int(getattr(params, "seed", 0)) + 402,
        )
        metadata["center_range_1_um"] = [center_min_1, center_max_1]
        metadata["center_range_2_um"] = [center_min_2, center_max_2]
        metadata["training_target_center_pool_1_path"] = center_pool_path_1
        metadata["training_target_center_pool_2_path"] = center_pool_path_2

    if getattr(params, "fix_q_evaluation_noise", False):
        q_eval_num_samples = max(1, int(getattr(params, "q_eval_num_samples", 0)))

        thickness_generator = _cpu_generator(int(getattr(params, "seed", 0)) + 201)
        thickness_noise = torch.randn(
            q_eval_num_samples,
            int(getattr(params, "thickness_noise_dim", 0)),
            generator=thickness_generator,
            dtype=torch.float32,
        )
        thickness_noise_path = os.path.join(reproducibility_dir, "q_eval_thickness_noise.pt")
        torch.save(thickness_noise, thickness_noise_path)

        material_generator = _cpu_generator(int(getattr(params, "seed", 0)) + 301)
        material_noise = torch.randn(
            q_eval_num_samples,
            int(getattr(params, "material_noise_dim", 0)),
            generator=material_generator,
            dtype=torch.float32,
        )
        material_noise_path = os.path.join(reproducibility_dir, "q_eval_material_noise.pt")
        torch.save(material_noise, material_noise_path)

        assets["q_eval_thickness_noise"] = thickness_noise
        assets["q_eval_material_noise"] = material_noise
        metadata["q_eval_thickness_noise_path"] = thickness_noise_path
        metadata["q_eval_material_noise_path"] = material_noise_path

    metadata_path = os.path.join(reproducibility_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    assets["metadata_path"] = metadata_path

    return assets
