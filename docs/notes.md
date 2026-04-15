# Notes

## 需求（2026-04-15）
在 `double` 分支中完成双窄带目标的首轮代码改造：训练目标切换为双峰、训练期核心评估改为双窗口 Q、推理筛样改为双峰目标，并同步更新文档。

## 涉及文件
- `model/Lorentzian/lorentzian_curves.py`
- `model/Lorentzian/__init__.py`
- `train.py`
- `train/trainer.py`
- `train/q_evaluator.py`
- `train/high_quality_solution_collector.py`
- `train/sample_saver.py`
- `utils/reproducibility.py`
- `infer.py`
- `inference/filtering.py`
- `inference/qfactor.py`
- `inference/inferer.py`
- `inference/visualization.py`
- `config/training_config.yaml`
- `config/inference_config.yaml`
- `tests/__init__.py`
- `tests/test_double_lorentzian.py`
- `tests/test_dual_metrics.py`
- `tests/test_double_inference.py`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
conda run -n oneday python -m unittest tests.test_double_lorentzian tests.test_dual_metrics tests.test_double_inference -v
conda run -n oneday python -m py_compile train.py train/trainer.py train/q_evaluator.py train/high_quality_solution_collector.py train/sample_saver.py utils/reproducibility.py infer.py inference/filtering.py inference/qfactor.py inference/inferer.py inference/visualization.py model/Lorentzian/lorentzian_curves.py
```

## Git
- branch: `double`
- commits:
  - `feat: add double target spectrum configuration`
  - `feat: replace training metrics with dual peak evaluation`
  - `feat: add double target inference screening`
  - `docs: document double target workflow`

## 需求（2026-04-15）
为双窄带目标改造写 implementation plan，明确文件拆分、测试入口、验证命令和提交粒度，作为后续代码实现的执行说明。

## 涉及文件
- `docs/superpowers/plans/2026-04-15-double-narrowband-implementation.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
- 未验证（本次仅新增实现计划与记录）

## Git
- branch: `double`
- commit: `git commit -m "docs: add double narrowband implementation plan"`

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
