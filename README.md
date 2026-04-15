# 薄膜光学谱 GAN（Spectral GAN）

## 项目简介

本项目使用 GAN 生成多层薄膜的层厚和材料概率，再通过传输矩阵法（TMM）计算反射/吸收谱，使生成谱形逼近目标 Lorentzian 曲线。当前 `double` 分支已将默认目标从单窄带替换为同宽同高的双窄带目标，并同步改造了训练期评估、推理筛样和 Q 报告。仓库还包含结构优化、Q-factor 计算、批量样本分析与优质解去重分析脚本。

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
  visualization.py
train.py
infer.py
requirements.txt
```

## 关键文件说明

- `train.py`：训练入口，读取 `config/training_config.yaml` 并创建训练输出目录。
- `train/trainer.py`：GAN 训练主循环，当前生成双窄带目标谱并与生成光谱做对抗训练。
- `train/q_evaluator.py`：训练期大规模生成样本，批量计算双窗口 `Q1/Q2/q_min_pair`、双峰目标误差，以及结构固定度/每层主导材料概率统计，并输出 CSV 和图。
- `train/high_quality_solution_collector.py`：按 `q_min_pair`、双峰 MSE、双峰峰值和材料概率阈值筛选优质解，保存光谱、结构 JSON 和汇总统计图。
- `utils/reproducibility.py`：设置全局 seed，生成固定训练 center 池和固定评估噪声池。
- `infer.py`：推理入口，先读 `config/inference_config.yaml`，再允许 CLI 覆盖同名参数。
- `inference/inferer.py`：批量生成样本、按双峰目标谱筛选 best samples、计算 Pareto front 和双窗口 Q-factor。
- `model/net.py`：生成器与判别器定义。
- `model/TMM/`：TMM 光学计算。
- `data/myindex.py`：材料 `.mat` 文件读取与折射率插值。
- `config/training_config.yaml`：训练所需结构、材料、光学、训练和可视化参数。
- `config/inference_config.yaml`：推理所需模型路径、训练配置路径、筛选参数和输出目录。

## 运行方法

安装依赖：

```bash
pip install -r requirements.txt
```

训练：

```bash
python train.py --config config/training_config.yaml --output_dir results/spectral_gan
```

推理：

```bash
python infer.py --target_center_1 3.8 --target_center_2 5.2
```

说明：

- `infer.py` 默认读取 `config/inference_config.yaml`。
- 当前 `config/inference_config.yaml` 里的 `model_path` 指向历史 `results\spectral_gan\run_20260122_004657\models\generator_final.pth`。
- 当前仓库未发现该 `results/` 目录，因此本地是否可直接运行推理，待确认。
- 当前推理目标为双窄带，需同时提供 `target_center_1` 和 `target_center_2`。

## 配置说明

- `config/training_config.yaml`
  - 分组：`structure`、`materials`、`optics`、`generator`、`training`、`optimizer`、`visualization`、`q_evaluation`、`high_quality_collection`、`reproducibility`
  - `optics` 当前使用双窄带目标字段：`lorentz_center_range_1`、`lorentz_center_range_2`、`min_peak_spacing`、`max_peak_spacing`、`peak_height_ratio`
  - `q_evaluation` 除了 Q/MSE 外，还会统计 `dominant_material_prob_threshold` 阈值下的结构固定度；该统计跟随当前 epoch 的 `alpha`
  - `q_evaluation` 当前主指标为双窗口 `Q1/Q2/q_min_pair` 与双峰 `double_lorentz_mse/rmse`
  - `q_evaluation` 还支持 FOM 评分参数 `fom_q_ref`、`fom_lorentz_width`、`fom_rmse_ref`、`fom_weight`；当前 FOM 基于 `q_min_pair` 与双峰目标 RMSE 计算
- `config/inference_config.yaml`
  - 关键字段：`model_path`、`config_path`、`output_dir`、`num_samples`、`infer_batch_size`、`alpha`、`target_center_1`、`target_center_2`、`target_width`、`center_region`、`weight_factor`、`best_samples`、`q_eval_window`

当前仓库未发现更复杂的配置继承系统，主要是 YAML 加载加 CLI 覆盖。
训练目前采用 full-pool epoch 语义：若启用固定训练 center 池，则一个 epoch 会完整遍历该池的全部 batch，`save_interval` 和 `q_evaluation.interval` 也按 full-pool epoch 计数。

## 输出说明

- 训练输出：`results/spectral_gan/run_YYYYMMDD_HHMMSS/`
  - 典型内容：`models/`、`samples/`、`samples/data/`、`training_metrics.png`
  - 分布图：`thickness_distribution_evolution_combined.png`、`merged_layers_distribution_evolution_combined.png`
  - Q/MSE 评估：`q_evaluation/`、`q_mse_evaluation_summary.csv`、`q_mse_evaluation_curves.png`、`q_mse_metrics_epoch_*.csv`、`global_max_q_curve.png`、`global_max_q_curve.csv`、`global_best_fom_curve.png`、`global_best_fom_curve.csv`
  - `q_mse_metrics_epoch_*.csv` 当前额外包含 `q1`、`q2`、`q_min_pair`、`double_lorentz_mse`、`double_lorentz_rmse`、`peak_wavelength_1_um`、`peak_wavelength_2_um`、`peak_absorption_1`、`peak_absorption_2`、`fwhm_1_um`、`fwhm_2_um`、`q_score`、`rmse_score`、`fom` 等列
  - `q_mse_evaluation_summary.csv` 当前额外包含 `mean_q1`、`mean_q2`、`mean_q_min_pair`、`mean_double_mse`、`mean_double_rmse`、`dual_valid_ratio`、`epoch_best_fom`、`global_max_q`、`global_best_fom`
  - `global_max_q_curve.csv` 当前导出列为 `epoch_max_q_min_pair` 与 `global_max_q_min_pair`，文件名保持不变
  - 全局最优样本追踪：`q_evaluation/global_best_samples/`，当出现新的 `global_max_q` 或 `global_best_fom` 时，会新建按 epoch 命名的文件，例如 `global_max_q_epoch_0020_structure.txt`、`global_max_q_epoch_0020_spectrum.csv`、`global_max_q_epoch_0020_spectrum.png`；`global_best_fom` 同理。四个全局曲线文件会继续保留。
  - 结构固定度统计：`q_evaluation/material_certainty_epoch_*.png`、`q_evaluation/material_certainty_curves.png`、`q_evaluation/material_certainty_layers_epoch_*.csv`、`q_evaluation/material_certainty_layer_history.csv`
  - 优质解收集：`high_quality_solutions/`、`summary/high_quality_solutions.csv`、`summary/high_quality_solution_distributions.png`、`epoch_*/epoch_*_sample_*/`
  - `high_quality_solutions.csv` 当前主字段为 `q1`、`q2`、`q_min_pair`、`double_lorentz_mse`、`peak_wavelength_1_um`、`peak_wavelength_2_um`、`peak_absorption_1`、`peak_absorption_2`、`fwhm_1_um`、`fwhm_2_um`
  - 可复现资产：`reproducibility/`、`training_target_center_pool_1.csv`、`training_target_center_pool_2.csv`、`q_eval_thickness_noise.pt`、`q_eval_material_noise.pt`
- 推理输出：`generated_samples/best_samples_YYYYMMDD_HHMMSS/`
  - 典型内容：`target_spectrum.xlsx`、`best_sample_*_absorption.xlsx`、`best_sample_*_structure.txt`、`best_samples_q.txt`、`pareto_front/`
  - `best_samples_q.txt` 与 `pareto_samples_q.txt` 当前输出双窗口 `Q1/Q2/Q_min_pair`

## 阅读顺序

1. `README.md`
2. `config/training_config.yaml`
3. `train.py`
4. `config/inference_config.yaml`
5. `infer.py`
6. 按任务再读 `train/trainer.py`、`inference/inferer.py`、`model/TMM/`、`data/myindex.py`
