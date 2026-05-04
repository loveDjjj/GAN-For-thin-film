# 命令手册（double 分支）

## 环境
```bash
conda activate oneday
```

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 训练（默认配置）
```bash
python train.py --config config/training_config.yaml --output_dir results/spectral_gan
```

## 推理（双峰目标，CLI 显式覆盖）
```bash
python infer.py --model_path results/spectral_gan/run_YYYYMMDD_HHMMSS/models/generator_final.pth --config_path config/training_config.yaml --output_dir generated_samples --num_samples 10000 --infer_batch_size 1024 --alpha 20 --target_center_1 3.8 --target_center_2 5.2 --target_width 0.02 --center_region 0.02 --weight_factor 10 --best_samples 8 --q_eval_window 0.05
```

## 推理（仅使用 YAML 配置）
```bash
python infer.py
```

## 单元测试（双峰主链）
```bash
python -m unittest tests.test_double_lorentzian tests.test_dual_metrics tests.test_double_inference -v
```

## 关键文件语法检查
```bash
python -m py_compile train.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py train/sample_saver.py utils/reproducibility.py infer.py inference/filtering.py inference/qfactor.py inference/inferer.py inference/visualization.py model/Lorentzian/lorentzian_curves.py
```

## Smoke：训练（1 epoch 快速检查）
```bash
python train.py --config tests/.tmp/smoke_training.yaml --output_dir results/smoke_double
```

## Smoke：推理（小样本快速检查）
```bash
python infer.py --model_path results/smoke_double/run_YYYYMMDD_HHMMSS/models/generator_final.pth --config_path tests/.tmp/smoke_training.yaml --output_dir generated_samples_smoke --num_samples 8 --infer_batch_size 4 --alpha 5 --target_center_1 3.8 --target_center_2 5.2 --target_width 0.02 --center_region 0.02 --weight_factor 10 --best_samples 2 --q_eval_window 0.05
```

## 清理 Smoke 产物
```powershell
if (Test-Path -LiteralPath 'generated_samples_smoke') { Remove-Item -LiteralPath 'generated_samples_smoke' -Recurse -Force }
if (Test-Path -LiteralPath 'tests/.tmp') { Remove-Item -LiteralPath 'tests/.tmp' -Recurse -Force }
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
git restore --source=HEAD data/__pycache__ model/Lorentzian/__pycache__ model/TMM/__pycache__ model/__pycache__ train/__pycache__ utils/__pycache__
```

