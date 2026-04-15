# Double Narrowband Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-narrowband target system with a shared-width, shared-height double-narrowband target across training, training-time evaluation, inference screening, and exported reports on branch `double`.

**Architecture:** Keep the generator and TMM forward path unchanged. Replace only the target-spectrum layer: generate double Lorentzian real samples during training, evaluate spectra with two target windows and a dual-peak global error metric, and screen inference results against a double-peak target using `q_min_pair` plus double-target weighted RMSE.

**Tech Stack:** Python, PyTorch, YAML, built-in `unittest`, pandas, matplotlib

---

## File Structure

- `config/training_config.yaml`
  - Replace single-target training optics fields with double-target center ranges and spacing controls.
- `config/inference_config.yaml`
  - Replace single-target inference fields with `target_center_1/2` plus shared width/window settings.
- `train.py`
  - Load new training config fields into `params`.
- `infer.py`
  - Load new inference config fields and expose them to the inference runner.
- `model/Lorentzian/lorentzian_curves.py`
  - Add double-target generation helpers for single spectrum, batched spectra, and random-center sampling.
- `train/trainer.py`
  - Swap single-target real-spectrum generation for double-target generation.
- `train/q_evaluator.py`
  - Replace single-peak Q/MSE/FOM evaluation with dual-window Q metrics and double-target RMSE/FOM.
- `train/high_quality_solution_collector.py`
  - Replace single-peak screening fields with dual-peak fields.
- `inference/inferer.py`
  - Build double-target spectra and feed dual-target screening / dual-Q reporting.
- `inference/filtering.py`
  - Compute double-window weighted RMSE.
- `inference/qfactor.py`
  - Compute `Q1/Q2/q_min_pair` within target windows.
- `inference/visualization.py`
  - Export dual-target fields and plots.
- `tests/test_double_lorentzian.py`
  - Validate double-target generation and config loading.
- `tests/test_dual_metrics.py`
  - Validate dual-window Q metrics and double-target RMSE/FOM.
- `tests/test_double_inference.py`
  - Validate dual-target inference screening and dual-Q reports.
- `README.md`
  - Update usage, config descriptions, and output semantics.
- `docs/notes.md`
  - Record implementation-plan writing.
- `docs/logs/2026-03.md`
  - Record implementation-plan writing.

### Task 1: Add Double-Target Configuration and Spectrum Generation

**Files:**
- Create: `tests/test_double_lorentzian.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/model/Lorentzian/lorentzian_curves.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/train.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/infer.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/config/training_config.yaml`
- Modify: `O:/Optics Code/GAN_for_thin_film/config/inference_config.yaml`

- [ ] **Step 1: Write the failing tests**

```python
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from train import load_parameters as load_train_parameters


class DoubleLorentzianTests(unittest.TestCase):
    def test_generate_double_lorentzian_curves_has_two_target_peaks(self):
        wavelengths = torch.linspace(3.0, 6.0, 1200)
        curve = generate_double_lorentzian_curves(
            wavelengths=wavelengths,
            width=0.02,
            center1=3.8,
            center2=5.1,
        )
        left_mask = (wavelengths >= 3.75) & (wavelengths <= 3.85)
        right_mask = (wavelengths >= 5.05) & (wavelengths <= 5.15)
        self.assertGreater(float(curve[left_mask].max()), 0.95)
        self.assertGreater(float(curve[right_mask].max()), 0.95)
        self.assertAlmostEqual(float(curve.max()), 1.0, places=5)

    def test_train_load_parameters_reads_double_target_fields(self):
        payload = {
            "structure": {"N_layers": 10, "pol": 0, "thickness_sup": 0.5, "thickness_bot": 0.05},
            "materials": {"materials_list": ["Ge", "SiO2"]},
            "optics": {
                "wavelength_range": [3, 6],
                "samples_total": 100,
                "theta": 0.0,
                "n_top": 1.0,
                "n_bot": 1.0,
                "lorentz_width": 0.02,
                "lorentz_center_range_1": [3.4, 4.2],
                "lorentz_center_range_2": [4.8, 5.6],
                "min_peak_spacing": 0.4,
                "max_peak_spacing": 2.5,
                "peak_height_ratio": 1.0,
                "metal_name": "Au",
            },
            "generator": {"thickness_noise_dim": 8, "material_noise_dim": 8, "alpha_min": 1, "alpha_max": 5},
            "training": {"epochs": 1, "batch_size": 2, "save_interval": 1, "noise_level": 0.0, "lambda_gp": 1.0},
            "optimizer": {"lr_gen": 1e-3, "lr_disc": 1e-4, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "training.yaml"
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            params = load_train_parameters(str(config_path), torch.device("cpu"))
        self.assertEqual(params.lorentz_center_range_1, [3.4, 4.2])
        self.assertEqual(params.lorentz_center_range_2, [4.8, 5.6])
        self.assertAlmostEqual(params.min_peak_spacing, 0.4)
        self.assertAlmostEqual(params.max_peak_spacing, 2.5)
        self.assertAlmostEqual(params.peak_height_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_double_lorentzian -v`
Expected: FAIL with import errors or missing `generate_double_lorentzian_curves` / missing double-target config attributes.

- [ ] **Step 3: Write the minimal implementation**

```python
# model/Lorentzian/lorentzian_curves.py
def _normalize_curves(curves):
    max_values = curves.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(curves.dtype).eps)
    return curves / max_values


def _sample_centers(batch_size, center_range, device):
    center_min, center_max = float(center_range[0]), float(center_range[1])
    return torch.empty(batch_size, dtype=torch.float32, device=device).uniform_(center_min, center_max)


def _enforce_peak_spacing(centers1, centers2, min_spacing, max_spacing):
    delta = (centers2 - centers1).abs().clamp_min(min_spacing)
    if max_spacing is not None:
        delta = delta.clamp_max(max_spacing)
    ordered_min = torch.minimum(centers1, centers2)
    return ordered_min, ordered_min + delta


def generate_double_lorentzian_curves(
    wavelengths,
    width,
    center1=None,
    center2=None,
    centers1=None,
    centers2=None,
    batch_size=None,
    center_range_1=None,
    center_range_2=None,
    min_peak_spacing=0.0,
    max_peak_spacing=None,
):
    if not torch.is_tensor(wavelengths):
        wavelengths = torch.tensor(wavelengths, dtype=torch.float32)
    wavelengths = wavelengths.float()
    device = wavelengths.device
    gamma = torch.tensor(width, dtype=torch.float32, device=device)

    if center1 is not None and center2 is not None:
        centers1 = torch.tensor([center1], dtype=torch.float32, device=device)
        centers2 = torch.tensor([center2], dtype=torch.float32, device=device)
    elif centers1 is not None and centers2 is not None:
        centers1 = torch.as_tensor(centers1, dtype=torch.float32, device=device).flatten()
        centers2 = torch.as_tensor(centers2, dtype=torch.float32, device=device).flatten()
    elif batch_size is not None:
        centers1 = _sample_centers(batch_size, center_range_1, device)
        centers2 = _sample_centers(batch_size, center_range_2, device)
    else:
        raise ValueError("provide (center1, center2), (centers1, centers2), or batch_size with center ranges")

    centers1, centers2 = _enforce_peak_spacing(centers1, centers2, min_peak_spacing, max_peak_spacing)
    peak1 = (gamma / 2) / ((wavelengths.unsqueeze(0) - centers1.unsqueeze(1)) ** 2 + (gamma / 2) ** 2)
    peak2 = (gamma / 2) / ((wavelengths.unsqueeze(0) - centers2.unsqueeze(1)) ** 2 + (gamma / 2) ** 2)
    curves = _normalize_curves(peak1 + peak2)
    if curves.shape[0] == 1 and center1 is not None and center2 is not None:
        return curves[0]
    return curves
```

```python
# train.py / infer.py inside load_parameters()
params.lorentz_center_range_1 = optics["lorentz_center_range_1"]
params.lorentz_center_range_2 = optics["lorentz_center_range_2"]
params.min_peak_spacing = optics["min_peak_spacing"]
params.max_peak_spacing = optics.get("max_peak_spacing")
params.peak_height_ratio = float(optics.get("peak_height_ratio", 1.0))
```

```yaml
# config/training_config.yaml optics section
lorentz_width: 0.02
lorentz_center_range_1: [3.2, 4.2]
lorentz_center_range_2: [4.6, 5.8]
min_peak_spacing: 0.4
max_peak_spacing: 2.5
peak_height_ratio: 1.0
```

```yaml
# config/inference_config.yaml
target_center_1: 3.8
target_center_2: 5.2
target_width: 0.02
center_region: 0.02
weight_factor: 10.0
q_eval_window: 0.05
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_double_lorentzian -v`
Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_double_lorentzian.py model/Lorentzian/lorentzian_curves.py train.py infer.py config/training_config.yaml config/inference_config.yaml
git commit -m "feat: add double target spectrum configuration"
```

### Task 2: Replace Training-Time Metrics with Dual-Peak Evaluation

**Files:**
- Create: `tests/test_dual_metrics.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/train/trainer.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/train/q_evaluator.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/train/high_quality_solution_collector.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

import torch

from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves
from train.q_evaluator import (
    compute_dual_q_metrics_torch,
    compute_double_lorentzian_mse_torch,
    compute_dual_fom_scores_torch,
)


class DualMetricTests(unittest.TestCase):
    def test_dual_q_metrics_returns_two_valid_q_values(self):
        wavelengths = torch.linspace(3.0, 6.0, 2000)
        spectra = generate_double_lorentzian_curves(wavelengths, width=0.02, center1=3.7, center2=5.1).unsqueeze(0)
        metrics = compute_dual_q_metrics_torch(
            wavelengths=wavelengths,
            absorption_spectra=spectra,
            target_center_1=3.7,
            target_center_2=5.1,
            half_window=0.08,
        )
        self.assertTrue(bool(metrics["dual_valid_mask"][0].item()))
        self.assertGreater(float(metrics["q1_values"][0]), 10.0)
        self.assertGreater(float(metrics["q2_values"][0]), 10.0)
        self.assertAlmostEqual(
            float(metrics["q_min_pair_values"][0]),
            min(float(metrics["q1_values"][0]), float(metrics["q2_values"][0])),
            places=5,
        )

    def test_double_lorentzian_mse_near_zero_for_matching_target(self):
        wavelengths = torch.linspace(3.0, 6.0, 2000)
        spectra = generate_double_lorentzian_curves(wavelengths, width=0.02, center1=3.7, center2=5.1).unsqueeze(0)
        mse_results = compute_double_lorentzian_mse_torch(
            wavelengths=wavelengths,
            absorption_spectra=spectra,
            peak_wavelengths_1=torch.tensor([3.7]),
            peak_wavelengths_2=torch.tensor([5.1]),
            width=0.02,
        )
        self.assertLess(float(mse_results["double_lorentz_mse_values"][0]), 1e-6)

    def test_dual_fom_uses_q_min_pair(self):
        scores = compute_dual_fom_scores_torch(
            q_min_pair_values=torch.tensor([200.0]),
            rmse_values=torch.tensor([0.01]),
            dual_valid_mask=torch.tensor([True]),
            q_ref=200.0,
            rmse_ref=0.05,
            weight=0.5,
        )
        self.assertGreater(float(scores["fom_values"][0]), 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_dual_metrics -v`
Expected: FAIL because dual metric helpers do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# train/q_evaluator.py
def compute_dual_q_metrics_torch(wavelengths, absorption_spectra, target_center_1, target_center_2, half_window, eps=1e-12):
    q1 = compute_q_factors_in_window_torch(wavelengths, absorption_spectra, target_center_1, half_window, prefix="q1")
    q2 = compute_q_factors_in_window_torch(wavelengths, absorption_spectra, target_center_2, half_window, prefix="q2")
    dual_valid_mask = q1["q1_valid_mask"] & q2["q2_valid_mask"]
    q_min_pair_values = torch.minimum(q1["q1_values"], q2["q2_values"])
    return {**q1, **q2, "dual_valid_mask": dual_valid_mask, "q_min_pair_values": q_min_pair_values}


def compute_double_lorentzian_mse_torch(wavelengths, absorption_spectra, peak_wavelengths_1, peak_wavelengths_2, width):
    target_curves = generate_double_lorentzian_curves(
        wavelengths=wavelengths,
        width=width,
        centers1=peak_wavelengths_1,
        centers2=peak_wavelengths_2,
    )
    mse_values = torch.mean((absorption_spectra - target_curves) ** 2, dim=1)
    rmse_values = torch.sqrt(mse_values.clamp_min(0.0))
    return {"double_lorentz_mse_values": mse_values, "double_lorentz_rmse_values": rmse_values}


def compute_dual_fom_scores_torch(q_min_pair_values, rmse_values, dual_valid_mask, q_ref, rmse_ref, weight):
    q_score_values = 1.0 - torch.exp(-torch.log(torch.tensor(2.0, device=rmse_values.device)) * q_min_pair_values / q_ref)
    rmse_score_values = torch.exp(-torch.log(torch.tensor(2.0, device=rmse_values.device)) * rmse_values / rmse_ref)
    fom_values = dual_valid_mask.to(rmse_values.dtype) * torch.pow(q_score_values.clamp(0.0, 1.0), weight) * torch.pow(rmse_score_values.clamp(0.0, 1.0), 1.0 - weight)
    return {"q_score_values": q_score_values, "rmse_score_values": rmse_score_values, "fom_values": fom_values}
```

```python
# train/trainer.py inside epoch batch generation
real_absorption = generate_double_lorentzian_curves(
    wavelengths=wavelengths,
    batch_size=current_batch_size,
    width=params.lorentz_width,
    center_range_1=params.lorentz_center_range_1,
    center_range_2=params.lorentz_center_range_2,
    min_peak_spacing=params.min_peak_spacing,
    max_peak_spacing=params.max_peak_spacing,
).float()
```

```python
# train/high_quality_solution_collector.py criteria usage
high_quality_mask = (
    (q_mse_results["q_min_pair_values"] > criteria["q_min"])
    & (q_mse_results["double_lorentz_mse_values"] < criteria["mse_max"])
    & (q_mse_results["peak_absorptions_1"] > criteria["peak_min"])
    & (q_mse_results["peak_absorptions_2"] > criteria["peak_min"])
    & (min_dominant_probs > criteria["dominant_material_prob_min"])
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_dual_metrics -v`
Expected: PASS with 3 tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_dual_metrics.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py
git commit -m "feat: replace training metrics with dual peak evaluation"
```

### Task 3: Update Inference Screening, Dual-Window Q, and Exports

**Files:**
- Create: `tests/test_double_inference.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/inference/inferer.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/inference/filtering.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/inference/qfactor.py`
- Modify: `O:/Optics Code/GAN_for_thin_film/inference/visualization.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

import torch

from inference.filtering import calculate_weighted_rmse, select_best_samples
from inference.qfactor import compute_dual_q_for_spectra
from model.Lorentzian.lorentzian_curves import generate_double_lorentzian_curves


class DoubleInferenceTests(unittest.TestCase):
    def test_dual_window_weighted_rmse_prefers_matching_double_peak(self):
        wavelengths = torch.linspace(3.0, 6.0, 1500)
        target = generate_double_lorentzian_curves(wavelengths, width=0.02, center1=3.8, center2=5.0)
        good = target.unsqueeze(0)
        bad = generate_double_lorentzian_curves(wavelengths, width=0.02, center1=3.8, center2=4.1).unsqueeze(0)
        spectra = torch.cat([bad, good], dim=0)
        indices, _ = select_best_samples(
            absorption_spectra=spectra,
            wavelengths=wavelengths,
            target=target,
            centers=(3.8, 5.0),
            region_width=0.05,
            weight_factor=10.0,
            num_best=1,
        )
        self.assertEqual(int(indices[0].item()), 1)

    def test_compute_dual_q_for_spectra_returns_q_min_pair(self):
        wavelengths = torch.linspace(3.0, 6.0, 1500)
        spectra = generate_double_lorentzian_curves(wavelengths, width=0.02, center1=3.8, center2=5.0).unsqueeze(0)
        results = compute_dual_q_for_spectra(wavelengths, spectra, center_1=3.8, center_2=5.0, half_window=0.08)
        self.assertTrue(bool(results["dual_valid_mask"][0].item()))
        self.assertGreater(float(results["q_min_pair_values"][0]), 10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_double_inference -v`
Expected: FAIL because inference functions still use single-target signatures.

- [ ] **Step 3: Write the minimal implementation**

```python
# inference/inferer.py
def create_double_target_lorentzian(wavelengths, center_1, center_2, width):
    return generate_double_lorentzian_curves(
        wavelengths=wavelengths,
        width=width,
        center1=center_1,
        center2=center_2,
    )
```

```python
# inference/filtering.py
def _build_weights(wavelengths, centers, region_width, weight_factor):
    wavelengths = _ensure_tensor(wavelengths, dtype=torch.float32).flatten()
    weights = torch.ones_like(wavelengths)
    for center in centers:
        region = (wavelengths >= center - region_width) & (wavelengths <= center + region_width)
        weights = torch.where(region, torch.full_like(weights, float(weight_factor)), weights)
    return wavelengths, weights


def select_best_samples(absorption_spectra, wavelengths, target, centers, region_width, weight_factor, num_best=4):
    rmse_values = compute_weighted_rmse_all(absorption_spectra, wavelengths, target, centers, region_width, weight_factor)
    best_rmse, best_indices = torch.topk(rmse_values, k=min(int(num_best), int(rmse_values.numel())), largest=False, sorted=True)
    return best_indices, best_rmse
```

```python
# inference/qfactor.py
def compute_dual_q_for_spectra(wavelengths, absorption_spectra, center_1, center_2, half_window):
    q1 = compute_q_for_spectra(wavelengths, absorption_spectra, center_1, half_window)
    q2 = compute_q_for_spectra(wavelengths, absorption_spectra, center_2, half_window)
    q_min_pair_values = torch.minimum(q1["q_values"], q2["q_values"])
    dual_valid_mask = q1["valid_mask"] & q2["valid_mask"]
    return {**q1, **{f"q2_{k}": v for k, v in q2.items()}, "q_min_pair_values": q_min_pair_values, "dual_valid_mask": dual_valid_mask}
```

```python
# inference/visualization.py exported structure fields
f.write(f"Q1: {q_metrics['q1']:.4f}\n")
f.write(f"Q2: {q_metrics['q2']:.4f}\n")
f.write(f"Q_min_pair: {q_metrics['q_min_pair']:.4f}\n")
f.write(f"Peak 1 Wavelength: {q_metrics['peak_wavelength_1']}\n")
f.write(f"Peak 2 Wavelength: {q_metrics['peak_wavelength_2']}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_double_inference -v`
Expected: PASS with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_double_inference.py inference/inferer.py inference/filtering.py inference/qfactor.py inference/visualization.py
git commit -m "feat: add double target inference screening"
```

### Task 4: Update Documentation and Run End-to-End Verification

**Files:**
- Modify: `O:/Optics Code/GAN_for_thin_film/README.md`
- Modify: `O:/Optics Code/GAN_for_thin_film/docs/notes.md`
- Modify: `O:/Optics Code/GAN_for_thin_film/docs/logs/2026-03.md`

- [ ] **Step 1: Update README usage and output descriptions**

```markdown
## 运行方法

训练：

```bash
python train.py --config config/training_config.yaml --output_dir results/spectral_gan
```

推理：

```bash
python infer.py --target_center_1 3.8 --target_center_2 5.2
```

说明：

- 训练目标已改为同宽同高双窄带 Lorentzian。
- 推理输出的 Q 报告包含 `Q1`、`Q2` 和 `Q_min_pair`。
- `q_evaluation` 主指标已改为双窗口局部 Q 和双峰整体误差。
```

- [ ] **Step 2: Update notes and monthly log**

```markdown
## 需求（2026-04-15）
在 `double` 分支中将单窄带目标整体替换为双窄带目标，并同步修改训练、评估、推理和导出字段。

## 涉及文件
- `model/Lorentzian/lorentzian_curves.py`
- `train.py`
- `train/trainer.py`
- `train/q_evaluator.py`
- `train/high_quality_solution_collector.py`
- `infer.py`
- `inference/inferer.py`
- `inference/filtering.py`
- `inference/qfactor.py`
- `inference/visualization.py`
- `config/training_config.yaml`
- `config/inference_config.yaml`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`
```

- [ ] **Step 3: Run repository verification**

Run:

```bash
python -m unittest tests.test_double_lorentzian tests.test_dual_metrics tests.test_double_inference -v
python -m py_compile train.py infer.py model/Lorentzian/lorentzian_curves.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py inference/inferer.py inference/filtering.py inference/qfactor.py inference/visualization.py
```

Expected:

- All unit tests PASS
- `py_compile` exits without syntax errors

- [ ] **Step 4: Run a smoke training epoch and a smoke inference pass**

Run:

```bash
python train.py --config config/training_config.yaml --output_dir results/smoke_double
$run = Get-ChildItem 'results/smoke_double' -Directory | Sort-Object Name -Descending | Select-Object -First 1
python infer.py --model_path (Join-Path $run.FullName 'models/generator_final.pth') --config_path config/training_config.yaml --output_dir generated_samples_smoke
```

Expected:

- Training completes at least one epoch with double-target real samples
- `q_evaluation` emits dual-peak fields
- Inference outputs best samples with double-target reports

- [ ] **Step 5: Commit**

```bash
git add README.md docs/notes.md docs/logs/2026-03.md
git commit -m "docs: document double target workflow"
```

## Self-Review

### Spec coverage

- Double-target config: covered by Task 1
- Double-target generation: covered by Task 1
- Training real-sample replacement: covered by Task 2
- Dual-window Q / `q_min_pair`: covered by Task 2 and Task 3
- Double-target MSE/FOM: covered by Task 2
- High-quality solution field replacement: covered by Task 2
- Inference target and screening replacement: covered by Task 3
- Documentation and verification: covered by Task 4

### Placeholder scan

- No unresolved placeholder markers remain.

### Type consistency

- Double-target config names are consistent: `lorentz_center_range_1/2`, `target_center_1/2`
- Dual metric names are consistent: `q1`, `q2`, `q_min_pair`, `double_lorentz_mse`, `double_lorentz_rmse`
- Shared width/window names are consistent across training and inference
