# Notes

## 需求（2026-04-16）
把文件名层也彻底双峰化，将 `global_max_q_curve.*` 和 `global_max_q_epoch_*` 统一改成 `global_max_q_min_pair_*`。

## 涉及文件
- `train/q_evaluator.py`
- `tests/test_dual_metrics.py`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
conda run -n oneday python -m unittest tests.test_dual_metrics -v
```

## Git
- branch: `double`
- commit: `git commit -m "refactor: align dual peak artifact filenames"`

## 需求（2026-04-16）
把训练期 summary 层里剩余的历史兼容列名彻底双峰化，移除 `mean_q/max_q/mean_mse/mean_rmse` 等旧键，统一改为 `*_q_min_pair` 与 `*_double_*` 语义。

## 涉及文件
- `train/q_evaluator.py`
- `train/trainer.py`
- `tests/test_dual_metrics.py`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
conda run -n oneday python -m unittest tests.test_dual_metrics tests.test_double_inference -v
conda run -n oneday python -m py_compile train/q_evaluator.py train/trainer.py train/high_quality_solution_collector.py
```

## Git
- branch: `double`
- commit: `git commit -m "refactor: fully rename dual peak summary fields"`

## 需求（2026-04-16）
继续收缩 `train/q_evaluator.py` 内部的单峰兼容别名，把 `q_values/mse_values/rmse_values/peak_wavelengths/peak_absorptions/fwhm/valid_mask` 从主结果字典层移除，只保留双峰主字段在主链上传递。

## 涉及文件
- `train/q_evaluator.py`
- `tests/test_dual_metrics.py`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
conda run -n oneday python -m unittest tests.test_dual_metrics tests.test_double_inference -v
conda run -n oneday python -m py_compile train/q_evaluator.py train/high_quality_solution_collector.py
```

## Git
- branch: `double`
- commit: `git commit -m "refactor: shrink dual metric compatibility aliases"`

## 需求（2026-04-16）
继续收口双峰导出层，移除 `q_mse_metrics_epoch_*.csv`、`high_quality_solutions.csv` 和部分结构文本中的单峰泛化别名字段，并让 `global_max_q_curve.csv` 明确使用 `q_min_pair` 语义列名。

## 涉及文件
- `train/q_evaluator.py`
- `train/high_quality_solution_collector.py`
- `tests/test_dual_metrics.py`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
conda run -n oneday python -m unittest tests.test_dual_metrics tests.test_double_inference -v
conda run -n oneday python -m py_compile train/q_evaluator.py train/high_quality_solution_collector.py inference/visualization.py
```

## Git
- branch: `double`
- commit: `git commit -m "refactor: remove legacy single-peak export aliases"`

## 需求（2026-04-16）
继续收口双峰主链的语义细节，让训练历史曲线、单 epoch 评估图和推理导出文件优先使用双峰字段与命名，不再强依赖单峰列名。

## 涉及文件
- `train/q_evaluator.py`
- `inference/visualization.py`
- `tests/test_dual_metrics.py`
- `tests/test_double_inference.py`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
conda run -n oneday python -m unittest tests.test_dual_metrics tests.test_double_inference -v
conda run -n oneday python -m py_compile train/q_evaluator.py inference/visualization.py
```

## Git
- branch: `double`
- commit: `git commit -m "refactor: align dual peak reporting semantics"`

## 需求（2026-04-16）
删除与当前双峰训练主链无关的独立辅助脚本，只保留当前双峰训练与推理主链所需代码，并同步清理 README 的相关入口说明。

## 涉及文件
- `calculate_q_factor.py`
- `optimize_structure.py`
- `analyze_gan_samoples.py`
- `analyze_high_quality_solutions.py`
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
- commit: `git commit -m "refactor: drop unused auxiliary scripts"`

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

## 需求（2026-05-04）：补充文档 command 手册，给出双峰训练主链常用命令
## 涉及文件
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 修改
- 在 `README.md` 新增 `Command Manual (double branch)` 章节。
- 补充并统一了双峰主链常用命令：
  - 环境与依赖安装
  - 训练命令
  - 双峰推理命令（CLI 覆盖）
  - 仅 YAML 推理命令
  - 单元测试命令
  - 关键文件语法检查命令
  - 训练/推理 smoke 命令
  - smoke 产物清理命令

## 验证
- 文档变更已写入，未执行训练/推理（本次仅文档更新）。

## 需求（2026-05-04）：将命令手册改为独立文档 command.md（不放在 README 末尾）
## 涉及文件
- `command.md`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 修改
- 新建 `command.md`，集中维护双峰训练主链常用命令。
- 从 `README.md` 移除上一轮追加的 `Command Manual (double branch)` 章节，避免主说明文档过长。
- 命令内容保持与上一轮一致：环境、依赖、训练、推理、测试、语法检查、smoke、清理。

## 验证
- 文档变更已写入，未执行训练/推理（本次仅文档调整）。

## 需求（2026-05-04）：命令手册独立为 command.md，并将 README/command 文本统一改为中文，修复 README 乱码
## 涉及文件
- `README.md`
- `command.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 修改
- 新建并维护独立中文命令手册：`command.md`。
- 从 README 中移除命令手册内容，README 仅保留项目说明、结构、关键文件、输出语义和阅读顺序。
- 重写 `README.md` 为 UTF-8 中文内容，修复原有乱码。
- `command.md` 内容全部改为中文，保留双峰训练主链常用命令。

## 验证
- 文档写入检查通过；本次仅文档修改，未执行训练/推理。

## 需求（2026-05-04）：高质量解按结构去重（10nm 粒度），同结构按 Q 优先、MSE 次优保留最佳；保留 1nm 厚度表示
## 涉及文件
- `train/high_quality_solution_collector.py`
- `tests/test_dual_metrics.py`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 修改
- 在 `train/high_quality_solution_collector.py` 新增合并结构键：
  - `merged_structure_1nm_key`（厚度按 1nm 四舍五入）
  - `merged_structure_10nm_key`（厚度按 10nm 四舍五入）
- `high_quality_solutions.csv` 新增上述两列。
- 汇总阶段按 `merged_structure_10nm_key` 去重，只保留最优样本：
  - 先按 `q_min_pair` 降序
  - 再按 `double_lorentz_mse` 升序
- 采样批处理中尽量保留 GPU 张量，仅在写文件前将单样本必要数据转到 CPU，减少不必要的数据搬运。

## 验证
```bash
conda run -n oneday python -m py_compile train/high_quality_solution_collector.py tests/test_dual_metrics.py
conda run -n oneday python -m unittest tests.test_dual_metrics.DualMetricTests.test_high_quality_registry_uses_dual_field_names tests.test_dual_metrics.DualMetricTests.test_high_quality_summary_dedup_by_10nm_key_with_q_then_mse_priority -v
```
