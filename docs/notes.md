# Notes

## 需求（2026-04-15）
为 `double` 分支补充“双窄带目标替换单窄带”的正式设计说明，明确训练目标、训练期评估、推理筛样和高质量样本筛选的双峰改造方案。

## 涉及文件
- `docs/superpowers/specs/2026-04-15-double-narrowband-design.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
- 未验证（本次仅新增设计文档与记录）

## Git
- branch: `double`
- commit: `git commit -m "docs: add double narrowband design spec"`

## 需求（2026-04-01）
在训练期 `q_evaluation` 中，给 FOM 增加独立的 RMSE 计算路线：

- 保留原有 `lorentz_mse/rmse`，继续用于 Q/MSE 评估图和原有统计。
- 新增 `q_evaluation.fom_lorentz_width` 配置，专门用于 FOM 的峰值对齐 Lorentzian 宽度。
- 基于该宽度重新生成 Lorentzian 目标，批量并行计算 `fom_lorentz_mse` 和 `fom_lorentz_rmse`。
- FOM 的 `RMSE_score` 改为使用这条新的 `fom_lorentz_rmse`，不再直接使用原来的 `lorentz_rmse`。
- 样本级 CSV、summary CSV 和全局最优结构 `txt` 需要把这套独立误差数据写出来。

## 涉及文件
- `train.py`
- `train/q_evaluator.py`
- `train/high_quality_solution_collector.py`
- `utils/config_loader.py`
- `config/training_config.yaml`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
python -m py_compile train/q_evaluator.py train/high_quality_solution_collector.py train.py utils/config_loader.py
git diff -- train.py train/q_evaluator.py train/high_quality_solution_collector.py utils/config_loader.py config/training_config.yaml README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: add dedicated fom rmse width to q evaluation"`
