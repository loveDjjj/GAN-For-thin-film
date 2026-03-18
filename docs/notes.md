# Notes

## 需求（2026-03-18）
训练期间按配置周期性生成大规模样本，在统计 Q 值的同时，基于峰值波长和 `lorentz_width` 生成 Lorentzian 目标谱并批量计算 MSE，输出统计图和分析数据。

## 验证
```bash
python -m py_compile train.py train/trainer.py train/q_evaluator.py utils/config_loader.py
git diff -- train.py train/trainer.py train/q_evaluator.py config/training_config.yaml README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: add lorentzian mse to training evaluation"`
