# 双窄带目标设计说明

## 背景

当前项目以单窄带 Lorentzian 目标为训练真样本，训练、训练期 `q_evaluation`、推理筛样和高质量样本收集都默认“全谱只需要一个主峰”。

本设计在 `double` 分支中直接替换该单窄带体系，目标变为“同宽同高、两个中心可配置、训练时在给定范围内采样”的双窄带目标体系。

## 已确认约束

- 分支：`double`
- 范围：在 `double` 分支中直接替换单窄带，不保留 `single/double` 配置切换
- 目标形式：双窄带、同宽、同高
- 泛化范围：半通用双峰模型，两个中心波长可配置，训练时仍限定在给定范围内随机采样
- 当前阶段：只做设计，不改生成器结构和 TMM 物理前向

## 目标

1. 将训练真样本从单峰 Lorentzian 替换为双峰 Lorentzian。
2. 将训练期评估从单峰 `Q/MSE/FOM` 替换为双峰局部 `Q1/Q2` 与双峰整体误差。
3. 将推理筛样从单峰目标筛选替换为双峰目标筛选。
4. 将高质量样本定义改为“双峰都成立”，避免一个峰好、另一个峰塌掉时被误判为好样本。

## 不改动范围

以下模块保持不变：

- `model/net.py`
- `model/TMM/optical_calculator.py`
- `model/TMM/TMM.py`

原因：

- 生成器当前负责输出层厚与材料概率，不依赖目标峰数。
- TMM 当前负责结构到光谱的可微物理前向，不依赖目标峰数。
- 双峰改造应集中在“目标定义、评估定义、推理筛样定义”这一层，避免扩大改动面。

## 配置设计

### 训练配置

在 `config/training_config.yaml` 中，替换当前单峰目标相关字段：

- 保留 `optics.lorentz_width`
  - 含义：双峰共享半高宽
- 新增 `optics.lorentz_center_range_1`
  - 含义：第一条峰中心的训练采样范围
- 新增 `optics.lorentz_center_range_2`
  - 含义：第二条峰中心的训练采样范围
- 新增 `optics.min_peak_spacing`
  - 含义：两峰最小间距，避免训练采样到几乎重合的双峰
- 可选新增 `optics.max_peak_spacing`
  - 含义：两峰最大间距，控制目标跨度
- 保留但固定 `optics.peak_height_ratio = 1.0`
  - 第一版只支持同高，字段可以保留为后续扩展位

### 推理配置

在 `config/inference_config.yaml` 中，替换单峰字段：

- `target_center` -> `target_center_1`, `target_center_2`
- `target_width` 保留
- 第一版保留共享窗口和共享权重：
  - `center_region`
  - `weight_factor`
  - `q_eval_window`

这样推理配置简单，和“同宽同高”的目标定义保持一致。

## 目标谱生成设计

文件：

- `model/Lorentzian/lorentzian_curves.py`

设计：

- 新增双峰目标生成函数，推荐命名为 `generate_double_lorentzian_curves`
- 继续保留单峰实现作为内部复用工具，但 `double` 分支中的训练和推理都调用双峰函数

支持三种调用模式：

1. 指定 `center1/center2`，生成单条双峰目标
2. 指定 `centers1/centers2`，生成一个 batch
3. 指定两个中心范围和 `batch_size`，随机采样一个 batch

生成逻辑：

1. 生成两个同宽单峰 Lorentzian
2. 将两者相加
3. 对最终曲线整体归一化到峰值 1

这样输出维度与当前单峰实现一致，仍然为：

- 单条：`[W]`
- 批量：`[B, W]`

## 训练流程设计

文件：

- `train.py`
- `train/trainer.py`

设计：

- `train.py` 负责装载新的双峰配置字段并写入 `params`
- `train/trainer.py` 中真实样本生成逻辑从单峰 Lorentzian 替换为双峰 Lorentzian

替换后每个 batch 的真样本流程为：

1. 为每个样本采样 `center1`
2. 为每个样本采样 `center2`
3. 检查并修正两峰间距，使其满足 `min_peak_spacing/max_peak_spacing`
4. 生成双峰目标光谱
5. 将该双峰目标作为判别器真样本

保持不变的内容：

- 生成器输入输出
- TMM 前向
- 判别器结构
- `BCEWithLogits + gradient penalty` 的对抗训练框架

## 训练期评估设计

文件：

- `train/q_evaluator.py`

当前问题：

- 现有逻辑默认全谱只有一个主峰
- 现有 `Q/MSE/FOM` 统计会把双峰样本错误压缩成单峰指标

改造目标：

- 把评估从“单峰全谱指标”替换为“双窗口局部指标 + 双峰整体误差”

### 局部峰与 Q 定义

对每个样本：

1. 在目标 `center1` 附近窗口内找局部峰，得到 `peak_1`, `Q1`, `FWHM1`
2. 在目标 `center2` 附近窗口内找局部峰，得到 `peak_2`, `Q2`, `FWHM2`
3. 两侧都找到合法峰和半高点时，该样本才记为 `dual_valid`

主 Q 指标定义：

- `q_min_pair = min(Q1, Q2)`

原因：

- 平均值会掩盖“一峰很好、一峰很差”的情况
- 双峰任务的核心是两个窄带都成立

### 双峰误差定义

对每个样本，基于检测到的两个峰位置构造一条峰值对齐的双峰 Lorentzian，再计算：

- `double_lorentz_mse`
- `double_lorentz_rmse`

该误差用于评估整条谱是否接近双峰目标，而不是只评估某一个峰。

### FOM 定义

FOM 由以下两部分构成：

- `Q_score`：基于 `q_min_pair`
- `RMSE_score`：基于 `double_lorentz_rmse`

这样 FOM 的物理意义是“双峰都足够窄，且整体双峰形状也足够像目标”。

### Summary 字段

建议替换为以下主字段：

- `mean_q1`
- `mean_q2`
- `mean_q_min_pair`
- `mean_double_mse`
- `mean_double_rmse`
- `dual_valid_ratio`
- `epoch_best_fom`

## 高质量样本收集设计

文件：

- `train/high_quality_solution_collector.py`

高质量样本筛选条件改为：

- `q1 > q_threshold`
- `q2 > q_threshold`
  - 或等价地 `q_min_pair > q_threshold`
- `double_lorentz_mse < mse_threshold`
- `peak_absorption_1 > peak_min`
- `peak_absorption_2 > peak_min`
- `min_dominant_material_probability > dominant_prob_threshold`

输出字段改为双峰版本：

- `q1`
- `q2`
- `q_min_pair`
- `peak_wavelength_1_um`
- `peak_wavelength_2_um`
- `peak_absorption_1`
- `peak_absorption_2`
- `fwhm_1_um`
- `fwhm_2_um`
- `double_lorentz_mse`

## 推理与筛样设计

文件：

- `infer.py`
- `inference/inferer.py`
- `inference/filtering.py`
- `inference/qfactor.py`
- `inference/visualization.py`

### 目标谱

推理目标由单峰替换为双峰：

- `target_center_1`
- `target_center_2`
- `target_width`

由双峰目标生成函数生成目标光谱。

### Best sample 排序

当前单峰 weighted RMSE 改为双窗口 weighted RMSE：

1. 默认全谱权重为 1
2. 在 `target_center_1 ± center_region` 内提高权重
3. 在 `target_center_2 ± center_region` 内提高权重
4. 计算整条谱对双峰目标的加权 RMSE

`best_samples` 的排序依据仍为整体误差最小，但误差语义已切换为双峰目标拟合误差。

### Pareto front

第一版保留现有二维定义：

- x 轴：double weighted RMSE
- y 轴：total thickness

不引入第三个维度，避免第一版分析复杂度过高。

### 推理 Q 报告

推理 Q 不再做全谱“找两个最高峰”，而改为目标窗口法：

1. 在 `target_center_1 ± q_eval_window` 内找局部峰并算 `Q1`
2. 在 `target_center_2 ± q_eval_window` 内找局部峰并算 `Q2`
3. 派生：
   - `q_min_pair`
   - `dual_valid`

推理导出报告建议包含：

- `q1`
- `q2`
- `q_min_pair`
- `peak_1`
- `peak_2`
- `fwhm_1`
- `fwhm_2`

## 文件级改动清单

必须改：

- `config/training_config.yaml`
- `config/inference_config.yaml`
- `train.py`
- `infer.py`
- `model/Lorentzian/lorentzian_curves.py`
- `train/trainer.py`
- `train/q_evaluator.py`
- `train/high_quality_solution_collector.py`
- `inference/inferer.py`
- `inference/filtering.py`
- `inference/qfactor.py`
- `inference/visualization.py`

文档同步：

- `README.md`
- `docs/notes.md`
- `docs/logs/2026-03.md`

尽量不改：

- `model/net.py`
- `model/TMM/optical_calculator.py`
- `model/TMM/TMM.py`

## 实施顺序

1. 目标谱与配置层
2. 训练与训练期评估层
3. 推理与筛样层
4. 文档与验证层

## 验证方案

第一阶段只要求端到端逻辑一致，不要求性能最优。

### 函数级验证

- 新配置能否成功加载
- 双峰目标函数输出是否正确
- 双窗口 Q 计算是否能对人工双峰样本返回合理 `Q1/Q2`

### 小批量训练验证

- `train.py` 能跑通配置装载
- `trainer.py` 能生成双峰真样本并完成至少 1 个 epoch
- `q_evaluator.py` 能输出双峰指标 CSV 与图

### 小规模推理验证

- `infer.py` 能生成双峰目标
- `best samples` 能按双峰误差筛选
- Q 报告中能输出 `q1/q2/q_min_pair`

## 风险与控制

### 风险 1：只学出一个峰

表现：

- 一个峰很尖、另一个峰塌掉，但平均指标仍不算太差

控制：

- 用 `q_min_pair` 作为主 Q 指标

### 风险 2：训练、评估、推理语义不一致

表现：

- 训练目标是双峰，但评估或筛样仍沿用单峰逻辑

控制：

- 统一以“双窗口局部峰 + 双峰整体误差”为标准

### 风险 3：导出字段残留单峰命名

表现：

- CSV、TXT、绘图标题语义混乱

控制：

- 统一替换为双峰字段命名，避免单峰字段继续作为主语义存在

## 成功标准

1. 训练至少能跑通 1 个 epoch
2. `q_evaluation` 输出双峰主指标
3. `infer.py` 能按双峰目标筛样
4. 导出报告能清晰体现两个峰的位置、Q 和误差

## 本期不做

- 不将两个目标中心作为条件输入显式喂给生成器
- 不支持双峰不同宽度
- 不支持双峰不同高度
- 不保留单峰与双峰的运行时切换开关
