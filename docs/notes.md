# Notes

## 需求（2026-03-18）
训练期间按 YAML 配置周期性生成 1000 个样本，批量并行计算 Q 值，统计平均 Q 并绘图，同时为 training_config.yaml 补详细注释。

## 验证
```bash
python -m py_compile train.py train/trainer.py train/q_evaluator.py utils/config_loader.py
git diff -- train.py train/trainer.py train/q_evaluator.py utils/config_loader.py config/training_config.yaml README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: add periodic q evaluation during training"`
