import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch


def _ensure_float_tensor(wavelengths):
    if not torch.is_tensor(wavelengths):
        return torch.tensor(wavelengths, dtype=torch.float32)
    return wavelengths.float()


def _normalize_curves(curves):
    max_values = curves.amax(dim=-1, keepdim=True).clamp_min(torch.finfo(curves.dtype).eps)
    return curves / max_values


def _resolve_center_range(wavelengths, width, center_range):
    wave_min = wavelengths.min().item()
    wave_max = wavelengths.max().item()
    padding = max((wave_max - wave_min) * 0.1, width * 2)
    default_min = wave_min + padding
    default_max = wave_max - padding

    if center_range is None:
        return default_min, default_max

    if not isinstance(center_range, (list, tuple)) or len(center_range) != 2:
        raise ValueError("center_range must be a list/tuple with two values [min, max]")

    center_min, center_max = float(center_range[0]), float(center_range[1])
    if center_min == center_max:
        raise ValueError("center_range min and max must be different")
    if center_min > center_max:
        center_min, center_max = center_max, center_min
    return center_min, center_max


def _sample_centers(batch_size, center_range, wavelengths, width, device):
    center_min, center_max = _resolve_center_range(wavelengths, width, center_range)
    return torch.empty(batch_size, dtype=torch.float32, device=device).uniform_(center_min, center_max)


def _enforce_peak_spacing(centers1, centers2, min_peak_spacing=0.0, max_peak_spacing=None):
    if centers1.shape != centers2.shape:
        raise ValueError("centers1 and centers2 must have the same shape")

    min_peak_spacing = max(0.0, float(min_peak_spacing))
    max_peak_spacing = None if max_peak_spacing is None else float(max_peak_spacing)

    ordered_min = torch.minimum(centers1, centers2)
    ordered_max = torch.maximum(centers1, centers2)
    spacing = ordered_max - ordered_min

    if min_peak_spacing > 0:
        spacing = spacing.clamp_min(min_peak_spacing)
    if max_peak_spacing is not None:
        if max_peak_spacing <= 0:
            raise ValueError("max_peak_spacing must be positive when provided")
        spacing = spacing.clamp_max(max_peak_spacing)

    return ordered_min, ordered_min + spacing


def generate_lorentzian_curves(
    wavelengths,
    batch_size=None,
    width=0.05,
    center=None,
    center_range=None,
    centers=None,
):
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
        centers: 指定的一组中心波长，[batch_size]，用于固定样本池模式

    Returns:
        洛伦兹曲线张量 [batch_size, len(wavelengths)] 或 [len(wavelengths)]

    Examples:
        # 随机中心模式（用于训练）
        curves = generate_lorentzian_curves(wavelengths, batch_size=64, width=0.05, center_range=[3.5, 7.5])

        # 指定中心模式（用于目标曲线）
        target = generate_lorentzian_curves(wavelengths, width=0.01, center=4.26)
    """
    # 确保 wavelengths 是浮点数张量
    wavelengths = _ensure_float_tensor(wavelengths)

    device = wavelengths.device

    # 指定单个中心模式
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

    # 指定一组中心模式
    if centers is not None:
        centers = torch.as_tensor(centers, dtype=torch.float32, device=device).flatten()
        if centers.numel() == 0:
            raise ValueError("centers must not be empty")

        gamma = torch.tensor(width, dtype=torch.float32, device=device)
        curves = (gamma / 2) / ((wavelengths.unsqueeze(0) - centers.unsqueeze(1)) ** 2 + (gamma / 2) ** 2)
        max_values = curves.max(dim=1, keepdim=True).values
        curves = torch.where(max_values > 0, curves / max_values, curves)
        return curves

    # 随机中心模式
    if batch_size is None:
        raise ValueError("必须指定batch_size（随机中心模式）或center/centers（指定中心模式）")

    # 创建一个批次的洛伦兹曲线
    # 从输入的wavelengths获取波长范围
    sampled_centers = _sample_centers(batch_size, center_range, wavelengths, width, device)
    return generate_lorentzian_curves(wavelengths, width=width, centers=sampled_centers)


def generate_double_lorentzian_curves(
    wavelengths,
    width=0.05,
    center1=None,
    center2=None,
    centers1=None,
    centers2=None,
    batch_size=None,
    center_range_1=None,
    center_range_2=None,
    min_peak_spacing=0.0,
    max_peak_spacing=None,
):
    """生成同宽同高的双峰 Lorentzian 目标曲线。"""
    wavelengths = _ensure_float_tensor(wavelengths)
    device = wavelengths.device
    gamma = torch.tensor(width, dtype=torch.float32, device=device)

    if center1 is not None and center2 is not None:
        centers1 = torch.tensor([center1], dtype=torch.float32, device=device)
        centers2 = torch.tensor([center2], dtype=torch.float32, device=device)
    elif centers1 is not None and centers2 is not None:
        centers1 = torch.as_tensor(centers1, dtype=torch.float32, device=device).flatten()
        centers2 = torch.as_tensor(centers2, dtype=torch.float32, device=device).flatten()
    elif batch_size is not None:
        centers1 = _sample_centers(batch_size, center_range_1, wavelengths, width, device)
        centers2 = _sample_centers(batch_size, center_range_2, wavelengths, width, device)
    else:
        raise ValueError(
            "必须指定(center1, center2)、(centers1, centers2) 或 "
            "(batch_size, center_range_1, center_range_2)"
        )

    if centers1.numel() == 0 or centers2.numel() == 0:
        raise ValueError("centers1 and centers2 must not be empty")

    centers1, centers2 = _enforce_peak_spacing(
        centers1,
        centers2,
        min_peak_spacing=min_peak_spacing,
        max_peak_spacing=max_peak_spacing,
    )

    peak1 = (gamma / 2) / ((wavelengths.unsqueeze(0) - centers1.unsqueeze(1)) ** 2 + (gamma / 2) ** 2)
    peak2 = (gamma / 2) / ((wavelengths.unsqueeze(0) - centers2.unsqueeze(1)) ** 2 + (gamma / 2) ** 2)
    curves = _normalize_curves(peak1 + peak2)

    if center1 is not None and center2 is not None:
        return curves[0]

    return curves
