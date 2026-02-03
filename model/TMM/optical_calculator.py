import os
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from model.TMM.TMM import TMM_solver
from data.myindex import index
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def calculate_reflection(thicknesses, refractive_indices, params, device):
    """
    计算薄膜结构的反射率

    Args:
        thicknesses: 各层厚度
        refractive_indices: 各层折射率
        params: 模型参数
        device: 运行设备(GPU/CPU)
        metal_name: 金属材料名称，默认为铝材料

    Returns:
        reflection: 反射率
    """
    # 动态加载金属材料数据 (mat 文件波长为 nm, 已在 index 内转换为 µm)
    n_metal = index(params.metal_name)
    batch_size = thicknesses.size(0)
    k = params.k.to(device)

    air_refractive = torch.ones((batch_size, 1, len(k)), dtype=torch.complex128, device=device)
    Al_refractive = torch.zeros((batch_size, 1, len(k)), dtype=torch.complex128, device=device)

    # 计算波长(µm)，直接用于插值
    lam_um = 2 * np.pi / k.cpu().numpy()

    for i in range(len(k)):
        n_value = n_metal(lam_um[i])
        n_tensor = torch.tensor(n_value, dtype=torch.complex128, device=device)
        Al_refractive[:, 0, i] = n_tensor
    
    # 组合所有层的折射率 (空气-材料层-铝-空气)
    refractive_indices = torch.cat((air_refractive, refractive_indices), 1)
    refractive_indices = torch.cat((refractive_indices, Al_refractive), 1)
    refractive_indices = torch.cat((refractive_indices, air_refractive), 1)
    
    # 设置层厚度
    # 为避免 inf 参与三角/指数运算导致数值异常，空气层厚度设为 0
    air_length = torch.zeros(batch_size, 1, device=device)
    Al_length = torch.ones(batch_size, 1, device=device) * 0.2  # 铝层厚度0.2微米
    
    # 组合所有层的厚度 (空气-材料层-铝-空气)
    thicknesses = torch.cat((air_length, thicknesses), 1)
    thicknesses = torch.cat((thicknesses, Al_length), 1)
    thicknesses = torch.cat((thicknesses, air_length), 1)
    
    # 转换类型并调整维度
    thicknesses = thicknesses.to(torch.complex128)
    thicknesses = thicknesses.unsqueeze(-1).repeat(1, 1, len(k))
    thicknesses = torch.permute(thicknesses, (0, 2, 1))

    # 数值健壮性检查，尽早暴露 NaN/Inf
    def _check_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).sum().item()
            raise ValueError(f"[NaNGuard] {name} contains {bad} non-finite values")

    _check_finite("thicknesses", thicknesses)
    _check_finite("refractive_indices", refractive_indices)
    
    # 计算反射率（TMM_solver返回(R, T)元组，我们只需要R）
    reflection, _ = TMM_solver(thicknesses, refractive_indices, k, params.theta, params.pol)

    return reflection
