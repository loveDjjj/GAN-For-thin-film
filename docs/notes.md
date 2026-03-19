# Notes

## 需求（2026-03-19）
训练期间在大规模 Q/MSE 评估时同步收集优质解，按 Q、MSE、峰值和材料概率阈值筛选样本，保存光谱、结构 JSON、CSV 和最终分布统计图。

## 验证
```bash
python -m py_compile train.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py utils/config_loader.py
git diff -- train.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py utils/config_loader.py config/training_config.yaml README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: collect high-quality solutions during training"`
