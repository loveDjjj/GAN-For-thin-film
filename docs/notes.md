# Notes

## 需求（2026-04-01）
在训练期 `q_evaluation` 中，新增实时的全局最优样本追踪保存：

- 当 `global_max_q` 刷新时，新建一个带 epoch 的结构 `txt`、原始光谱 `csv` 和光谱 `png`。
- 当 `global_best_fom` 刷新时，新建一个带 epoch 的结构 `txt`、原始光谱 `csv` 和光谱 `png`。
- 不覆盖旧记录；每次出现新的全局最优，都在 `global_best_samples/` 下新增一组文件。
- 光谱 CSV 只保留原始光谱主数据，即 `wavelength_um` 和对应的 `absorption`。
- `global_best_fom_curve.csv/.png`、`global_max_q_curve.csv/.png` 这四个全局曲线文件继续保留。
- 复用现有 `q_evaluation` 结果中的厚度、材料概率和吸收光谱，不额外增加一次生成与 TMM 计算。

## 涉及文件
- `train/q_evaluator.py`
- `train/high_quality_solution_collector.py`
- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 验证
```bash
python -m py_compile train/q_evaluator.py train/high_quality_solution_collector.py train.py utils/config_loader.py
git diff -- train/q_evaluator.py train/high_quality_solution_collector.py README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: save per-update global best q-eval files"`
