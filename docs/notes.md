# Notes

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

## 需求（2026-05-12）：对比本地 data/Materials 与 refractiveindex.info 数据库的 n/k 曲线
## 涉及文件
- `compare_material_nk_sources.py`
- `docs/notes.md`
- `docs/logs/2026-03.md`

## 修改
- 新增独立脚本 `compare_material_nk_sources.py`。
- 对比材料与数据源：
  - Ge: 本地 `Ge.mat` vs `Ge:Amotchkina-film`
  - SiO2: 本地 `SiO2 (Glass) - Palik.mat` vs `SiO2:Kischkat`
  - ZnO: 本地 `ZnO.mat` vs `ZnO:Aguilar`
  - YbF3: 本地 `YbF3.mat` vs `YbF3:Amotchkina`
  - Si: 本地 `Si (Silicon) - Palik.mat` vs `Si:Shkondin`
- 波长范围默认 `3-6 um`，输出图和 CSV 使用 `nm` 为横轴单位。
- 每个材料输出一张 n/k 对比图和一个 CSV，同时输出 `nk_comparison_summary.json`。
- 脚本支持 `--db_path` 指向已克隆的 `refractiveindex.info-database/database`，避免自动下载数据库失败。

## 验证
```bash
conda run -n oneday python -m py_compile compare_material_nk_sources.py
```
- 当前本机可导入 `refractiveindex` 包，但首次自动下载数据库时被临时目录/下载权限阻止；需要使用本地克隆数据库目录运行。
