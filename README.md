# 薄膜光学谱 GAN（Spectral GAN）

本项目使用 GAN 生成薄膜结构参数（层厚与材料概率），通过并行传输矩阵法
（TMM）计算反射/吸收光谱，使生成谱形逼近目标洛伦兹吸收曲线。

## 核心思路
- 生成器：输出每层厚度与材料概率。
- TMM：将结构参数映射到光谱。
- 判别器：区分真实洛伦兹谱与生成谱。

## 入口脚本
- 训练入口：`train.py`
- 推理入口：`infer.py`

保留旧入口用于兼容：
- `train_material_combinations.py` -> 调用 `train.py`
- `load_gan_model.py` -> 调用 `infer.py`

## 主要结构
- `train.py`：训练入口，读取配置与参数。
- `train/trainer.py`：训练循环（GAN 优化）。
- `train/sample_saver.py`：训练过程样本可视化与保存。
- `infer.py`：推理入口，读取配置与模型。
- `inference/inferer.py`：推理流程（生成、评分、筛选）。
- `inference/results.py`：推理结果保存与可视化。
- `model/net.py`：生成器与判别器网络。
- `model/Lorentzian/lorentzian_curves.py`：洛伦兹谱生成器。
- `model/TMM/optical_calculator.py`：并行 TMM 反射计算。
- `model/TMM/TMM.py`：TMM 核心求解器。
- `data/myindex.py`：材料数据库与插值。
- `config/training_config.yaml`：训练配置（必需）。
- `config/inference_config.yaml`：推理配置（必需）。

## 安装
```bash
pip install -r requirements.txt
```

## 训练
```bash
python train.py --config config/training_config.yaml --output_dir results/spectral_gan
```

## 推理
```bash
python infer.py
```
默认读取 `config/inference_config.yaml`，命令行参数会覆盖 YAML。

## 配置说明
所有参数必须来自 YAML，不再存在代码内默认值。

### 训练配置：`config/training_config.yaml`
必需字段：
- `structure`：`N_layers`, `pol`, `thickness_sup`, `thickness_bot`
- `materials`：`materials_list`
- `optics`：`wavelength_range`, `samples_total`, `theta`, `n_top`, `n_bot`,
  `lorentz_width`, `lorentz_center_range`, `metal_name`
- `generator`：`thickness_noise_dim`, `material_noise_dim`, `alpha_sup`, `alpha`
- `training`：`epochs`, `batch_size`, `save_interval`, `noise_level`, `lambda_gp`
- `optimizer`：`lr_gen`, `lr_disc`, `beta1`, `beta2`, `weight_decay`

### 推理配置：`config/inference_config.yaml`
必需字段：
- `model_path`, `config_path`, `output_dir`, `num_samples`, `alpha`
- `target_center`, `target_width`, `center_region`, `weight_factor`, `best_samples`

## 输出说明
训练输出（每次运行一个目录）：
- `models/`：生成器/判别器权重
- `samples/`：真实谱与生成谱对比图
- `samples/data/`：谱数据与结构文本
- `training_metrics.png`：训练曲线

推理输出：
- `generated_samples/best_samples_*/`：筛选出的谱、结构与对比图

## 备注
- 洛伦兹真实谱在 GPU 上批量生成。
- 衬底金属材料由 `optics.metal_name` 配置。
