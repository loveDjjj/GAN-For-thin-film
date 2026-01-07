import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch


def generate_lorentzian_curves(wavelengths, batch_size=None, width=0.05, center=None, center_range=None):
    """
    生成洛伦兹曲线作为真实吸收率曲线

    支持两种模式：
    1. 随机中心模式：指定batch_size，生成多个随机中心的曲线
    2. 指定中心模式：指定center，生成单个指定中心的曲线

    Args:
        wavelengths: 波长数组（numpy array或torch tensor）
        batch_size: 批次大小（随机中心模式）
        width: 洛伦兹曲线带宽（微米）
        center: 指定的中心波长（微米，指定中心模式）

    Returns:
        洛伦兹曲线张量 [batch_size, len(wavelengths)] 或 [len(wavelengths)]

    Examples:
        # 随机中心模式（用于训练）
        curves = generate_lorentzian_curves(wavelengths, batch_size=64, width=0.05, center_range=[3.5, 7.5])

        # 指定中心模式（用于目标曲线）
        target = generate_lorentzian_curves(wavelengths, width=0.01, center=4.26)
    """
    # 确保 wavelengths 是浮点数张量
    if not torch.is_tensor(wavelengths):
        wavelengths = torch.tensor(wavelengths, dtype=torch.float32)
    else:
        wavelengths = wavelengths.float()

    device = wavelengths.device

    # 指定中心模式
    if center is not None:
        gamma = torch.tensor(width, dtype=torch.float32, device=device)
        center_tensor = torch.tensor(center, dtype=torch.float32, device=device)

        # 洛伦兹分布公式
        curve = (gamma/2) / ((wavelengths - center_tensor)**2 + (gamma/2)**2)

        # 标准化曲线，最大值为1
        max_val = torch.max(curve)
        if max_val > 0:
            curve = curve / max_val

        return curve

    # 随机中心模式
    if batch_size is None:
        raise ValueError("必须指定batch_size（随机中心模式）或center（指定中心模式）")

    # 创建一个批次的洛伦兹曲线
    curves = torch.zeros(batch_size, len(wavelengths), dtype=torch.float32, device=device)

    # 从输入的wavelengths获取波长范围
    wave_min = wavelengths.min().item()
    wave_max = wavelengths.max().item()

    # 在实际波长范围内生成中心，稍微缩小范围确保峰值可见
    # 根据带宽参数动态调整填充范围，确保整个峰值在可见范围内
    # 带宽越大，需要的填充越大，以确保峰值完全可见
    padding = max((wave_max - wave_min) * 0.1, width * 2)  # 使用带宽的2倍或10%范围的较大值
    centers = torch.FloatTensor(batch_size).uniform_(
        wave_min + padding,
        wave_max - padding
    ).to(device)

    # 计算每个样本的洛伦兹曲线
    for i in range(batch_size):
        center_i = centers[i]
        # 洛伦兹分布公式: L(x) = (1/π) * (γ/2) / ((x - x₀)² + (γ/2)²)
        # 其中γ是带宽，x₀是中心位置
        gamma = torch.tensor(width, dtype=torch.float32, device=device)
        curves[i] = (gamma/2) / ((wavelengths - center_i)**2 + (gamma/2)**2)

        # 标准化曲线，最大值为1
        max_val = torch.max(curves[i])
        if max_val > 0:
            curves[i] = curves[i] / max_val

    return curves