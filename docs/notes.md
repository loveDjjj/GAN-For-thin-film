# Notes

## 需求（2026-03-25）
新增一个脚本，用于读取 `high_quality_solutions.csv`，把优质解结构转成整数 nm 的材料序列后去重，并输出去重 CSV 与波长-Q 颜色散点图。

## 验证
```bash
python -m py_compile analyze_high_quality_solutions.py
git diff -- analyze_high_quality_solutions.py README.md docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: add high-quality solution dedup and plotting script"`
