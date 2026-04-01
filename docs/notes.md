# Notes

## 需求（2026-04-01）
在训练期 `q_evaluation` 中，新增实时的全局最优样本追踪保存：

- 当 `global_max_q` 刷新时，追加保存该样本的汇总信息、逐层结构信息、合并层结构信息和光谱信息，全部为 CSV。
- 当 `global_best_fom` 刷新时，追加保存该样本的汇总信息、逐层结构信息、合并层结构信息和光谱信息，全部为 CSV。
- 以上历史记录都要保留，不覆盖旧记录；只有出现新的全局最优时才在 CSV 末尾追加。
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
- commit: `git commit -m "feat: persist global best q-eval sample histories"`
