# Notes

## 需求（2026-03-19）
将训练循环改成 full-pool epoch：每个 epoch 跑完整个固定 center 池的全部 batch，d_steps/g_steps 作为每个 batch 的更新比例，并同步调整 epoch 相关配置语义和值。

## 验证
```bash
python -m py_compile train.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py utils/config_loader.py utils/reproducibility.py model/Lorentzian/lorentzian_curves.py
git diff -- train.py train/trainer.py train/q_evaluator.py utils/config_loader.py utils/reproducibility.py model/Lorentzian/lorentzian_curves.py config/training_config.yaml README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "refactor: switch training to full-pool epochs"`
