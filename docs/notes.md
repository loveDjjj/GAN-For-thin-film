# Notes

## 需求（2026-03-18）
将训练输出中的厚度分布图和合并层分布图分别改为单张 2x2 汇总图，不再保留原始单图输出。

## 验证
```bash
python -m py_compile utils/visualize.py
git diff -- utils/visualize.py README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: combine distribution evolution plots into 2x2 overviews"`
