# 薄膜光学谱 GAN（Spectral GAN）

## 项目简介

本项目使用 GAN 生成多层薄膜的层厚和材料概率，再通过传输矩阵法（TMM）计算反射/吸收谱，使生成谱形逼近目标 Lorentzian 曲线。当前仓库还包含推理筛选、Q-factor 计算、结构优化、批量样本分析，以及训练期批量 Q/MSE 评估。

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
inference/
  inferer.py
  filtering.py
  visualization.py
train.py
infer.py
calculate_q_factor.py
optimize_structure.py
analyze_gan_samoples.py
requirements.txt
```

## 关键文件说明

- `train.py`：训练入口，读取 `config/training_config.yaml` 并创建训练输出目录。
- `train/trainer.py`：GAN 训练主循环，调用 Lorentzian 目标谱和 TMM 反射率计算。
- `train/q_evaluator.py`：训练期大规模生成样本，批量计算每个样本的 Q 值和峰值对齐 Lorentzian MSE，并输出统计结果。
- `train/high_quality_solution_collector.py`：按 Q、MSE、峰值和材料概率阈值筛选优质解，保存光谱、结构 JSON 和汇总统计图。
- `infer.py`：推理入口，先读 `config/inference_config.yaml`，再允许 CLI 覆盖同名参数。
- `inference/inferer.py`：批量生成样本、按目标谱筛选 best samples、计算 Pareto front 和 Q-factor。
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
python infer.py
```

说明：

- `infer.py` 默认读取 `config/inference_config.yaml`。
- 当前 `config/inference_config.yaml` 里的 `model_path` 指向历史 `results\spectral_gan\run_20260122_004657\models\generator_final.pth`。
- 当前仓库未发现该 `results/` 目录，因此本地是否可直接运行推理，待确认。

Q-factor 计算：

```bash
python calculate_q_factor.py --file generated_samples/<run>/best_sample_1_absorption.xlsx --center 4.26 --range 0.1 --output_dir q_results --plot
```

结构优化：

```bash
python optimize_structure.py --file generated_samples/<run>/best_sample_1_structure.txt --center1 4.26 --center2 8.5 --range 0.3 --output_dir optimized_results
```

批量样本分析：

```bash
python analyze_gan_samoples.py --model_path <generator_final.pth> --config_path config/training_config.yaml --output_dir analysis_results --max_samples 100000 --batch_size 500 --alpha 200
```

## 配置说明

- `config/training_config.yaml`
  - 分组：`structure`、`materials`、`optics`、`generator`、`training`、`optimizer`、`visualization`、`q_evaluation`、`high_quality_collection`
- `config/inference_config.yaml`
  - 关键字段：`model_path`、`config_path`、`output_dir`、`num_samples`、`infer_batch_size`、`alpha`、`target_center`、`target_width`、`center_region`、`weight_factor`、`best_samples`、`q_eval_window`

当前仓库未发现更复杂的配置继承系统，主要是 YAML 加载加 CLI 覆盖。

## 输出说明

- 训练输出：`results/spectral_gan/run_YYYYMMDD_HHMMSS/`
  - 典型内容：`models/`、`samples/`、`samples/data/`、`training_metrics.png`
  - 分布图：`thickness_distribution_evolution_combined.png`、`merged_layers_distribution_evolution_combined.png`
  - Q/MSE 评估：`q_evaluation/`、`q_mse_evaluation_summary.csv`、`q_mse_evaluation_curves.png`、`q_mse_metrics_epoch_*.csv`
  - 优质解收集：`high_quality_solutions/`、`summary/high_quality_solutions.csv`、`summary/high_quality_solution_distributions.png`、`epoch_*/epoch_*_sample_*/`
- 推理输出：`generated_samples/best_samples_YYYYMMDD_HHMMSS/`
  - 典型内容：`best_sample_*_absorption.xlsx`、`best_sample_*_structure.txt`、`best_samples_q.txt`、`pareto_front/`
- 其他脚本输出：
  - `calculate_q_factor.py` -> `q_results/`
  - `optimize_structure.py` -> `optimized_results/`
  - `analyze_gan_samoples.py` -> `analysis_results/`

## 阅读顺序

1. `README.md`
2. `config/training_config.yaml`
3. `train.py`
4. `config/inference_config.yaml`
5. `infer.py`
6. 按任务再读 `train/trainer.py`、`inference/inferer.py`、`model/TMM/`、`data/myindex.py`
