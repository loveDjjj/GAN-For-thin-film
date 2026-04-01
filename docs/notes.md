# Notes

## 需求（2026-04-01）
在训练期 `q_evaluation` 统计中新增两条不影响原有图表的历史最优统计路线：

- 统计并绘制 `global_max_q`，表示截至当前 epoch 的全局最大 Q 随训练的累计变化。
- 定义并统计 `FOM`，其中：
  - `RMSE = sqrt(lorentz_mse)`
  - `Q_score = 1 - exp(-ln(2) * Q / Q_ref)`
  - `RMSE_score = exp(-ln(2) * RMSE / RMSE_ref)`
  - `FOM = valid * (Q_score ^ w) * (RMSE_score ^ (1 - w))`
- 统计并绘制 `global_best_fom`，表示截至当前 epoch 的全局最佳 FOM 随训练的累计变化。
- 保存新增曲线对应的 CSV 数据，并将新增配置项与输出说明同步到文档。

## 涉及文件
- `train.py`
- `train/q_evaluator.py`
- `utils/config_loader.py`
- `config/training_config.yaml`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
python -m py_compile train.py train/q_evaluator.py utils/config_loader.py
git diff -- train.py train/q_evaluator.py utils/config_loader.py config/training_config.yaml README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: add global q and fom tracking to q evaluation"`
