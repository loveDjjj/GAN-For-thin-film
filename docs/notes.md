# Notes

## 需求（2026-03-24）
优化反射率计算和推理筛样链路：去掉每次反射计算里的金属材料重复读取/CPU 插值/逐波长 Python 循环，并把推理阶段的筛样、Q 计算、top-k 尽量留在 GPU，最后再搬运保存所需样本。

## 验证
```bash
python -m py_compile train.py infer.py model/TMM/optical_calculator.py inference/filtering.py inference/qfactor.py inference/inferer.py inference/visualization.py
git diff -- train.py infer.py model/TMM/optical_calculator.py inference/filtering.py inference/qfactor.py inference/inferer.py inference/visualization.py docs/notes.md docs/logs/2026-03.md
```

## Git
- branch: `main`
- commit: `git commit -m "feat: accelerate reflection and inference screening on gpu"`
