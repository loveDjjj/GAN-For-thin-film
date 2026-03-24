# Notes

## 需求（2026-03-24）
将当前 main 分支的 TMM 金属层计算逻辑改回与 master 一致，同时保留一次缓存到 GPU 的实现方式，并统一使用 complex128。

## 验证
```bash
python -m py_compile train.py infer.py model/TMM/optical_calculator.py
git diff -- train.py infer.py model/TMM/optical_calculator.py docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "fix: restore master-compatible tmm metal interpolation"`
