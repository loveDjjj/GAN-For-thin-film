# 薄膜光学谱 GAN（double 分支）

## 项目简介
本项目使用 GAN 生成多层薄膜结构（层厚与材料概率），并通过传输矩阵法（TMM）计算反射/吸收光谱。  
当前 `double` 分支已将主目标改为**双窄带（同宽同高）**，并同步改造了训练期评估、推理筛样与 Q 报告语义。

## 项目结构
```text
config/
  training_config.yaml
  inference_config.yaml
data/
  Materials/
  myindex.py
model/
  net.py
  Lorentzian/
  TMM/
train/
  trainer.py
  q_evaluator.py
  high_quality_solution_collector.py
  sample_saver.py
utils/
  reproducibility.py
inference/
  inferer.py
  filtering.py
  qfactor.py
  visualization.py
tests/
train.py
infer.py
command.md
requirements.txt
```

## 关键文件
- `train.py`：训练入口，加载训练配置并创建输出目录。
- `train/trainer.py`：GAN 训练主循环，构建双峰目标并执行对抗训练。
- `train/q_evaluator.py`：训练期批量评估，输出 `Q1/Q2/q_min_pair` 与双峰误差指标。
- `train/high_quality_solution_collector.py`：按双峰阈值筛选并保存高质量结构。
- `infer.py`：推理入口，支持 YAML 配置与 CLI 覆盖。
- `inference/inferer.py`：批量采样、双峰目标筛样、Pareto 前沿与 Q 报告。
- `model/net.py`：生成器/判别器定义。
- `model/TMM/`：光学传输矩阵计算。
- `config/training_config.yaml`：训练参数与双峰目标采样范围。
- `config/inference_config.yaml`：推理参数与双峰目标配置。
- `command.md`：常用命令手册（训练/推理/测试/smoke）。

## 运行说明
常用命令已统一放入：
- `command.md`

建议阅读顺序：
1. `README.md`
2. `command.md`
3. `config/training_config.yaml`
4. `train.py`
5. `config/inference_config.yaml`
6. `infer.py`

## 输出说明（双峰语义）
- 训练输出：`results/spectral_gan/run_YYYYMMDD_HHMMSS/`
  - `q_evaluation/q_mse_metrics_epoch_*.csv`：包含 `q1/q2/q_min_pair/double_lorentz_mse/double_lorentz_rmse` 等字段。
  - `q_evaluation/q_mse_evaluation_summary.csv`：包含 `mean_q_min_pair/global_max_q_min_pair/global_best_fom` 等字段。
  - `q_evaluation/global_max_q_min_pair_curve.csv/.png`
  - `q_evaluation/global_best_samples/global_max_q_min_pair_epoch_*.{txt,csv,png}`
- 推理输出：`generated_samples/best_samples_YYYYMMDD_HHMMSS/`
  - `target_spectrum.xlsx`
  - `best_samples_q.txt`、`pareto_samples_q.txt`：包含 `Q1/Q2/Q_min_pair`

## 待确认
- 默认 `config/inference_config.yaml` 中的 `model_path` 是否指向你当前机器上存在的模型文件，需要按本地实际结果目录确认。

