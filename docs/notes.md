# Notes

## 需求（2026-03-24）
在训练期 q_evaluation 中增加结构固定度统计：保存每个结构各层的主导材料概率、统计 fully fixed 结构占比，并输出对应 CSV 与图。

## 验证
```bash
python -m py_compile train.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py utils/config_loader.py utils/reproducibility.py model/Lorentzian/lorentzian_curves.py
git diff -- train.py train/trainer.py train/q_evaluator.py utils/config_loader.py config/training_config.yaml README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: add material certainty stats to q evaluation"`
