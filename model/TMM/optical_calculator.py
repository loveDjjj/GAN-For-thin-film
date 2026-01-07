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
    # 动态加载金属材料数据
    n_metal = index(params.metal_name)
    # 获取batch_size和波数
    batch_size = thicknesses.size(0)
    k = params.k.to(device)
    
    # 创建空气和铝层折射率
    air_refractive = torch.ones((batch_size, 1, len(k)), dtype=torch.complex128, device=device)
    Al_refractive = torch.zeros((batch_size, 1, len(k)), dtype=torch.complex128, device=device)
    
    # 计算波长(微米)
    lam = 1000 * 2 * np.pi / k.cpu().numpy()
    
    # 修复：将numpy值正确转换为torch张量
    for i in range(len(k)):
        # 获取金属的折射率（复数形式）
        n_value = n_metal(lam[i])
        # 转换为torch复数张量并移至正确设备
        n_tensor = torch.tensor(n_value, dtype=torch.complex128, device=device)
        # 为每个批次赋值
        Al_refractive[:, 0, i] = n_tensor
    
    # 组合所有层的折射率 (空气-材料层-铝-空气)
    refractive_indices = torch.cat((air_refractive, refractive_indices), 1)
    refractive_indices = torch.cat((refractive_indices, Al_refractive), 1)
    refractive_indices = torch.cat((refractive_indices, air_refractive), 1)
    
    # 设置层厚度
    air_length = torch.ones(batch_size, 1, device=device) * float('inf')
    Al_length = torch.ones(batch_size, 1, device=device) * 0.2  # 铝层厚度0.2微米
    
    # 组合所有层的厚度 (空气-材料层-铝-空气)
    thicknesses = torch.cat((air_length, thicknesses), 1)
    thicknesses = torch.cat((thicknesses, Al_length), 1)
    thicknesses = torch.cat((thicknesses, air_length), 1)
    
    # 转换类型并调整维度
    thicknesses = thicknesses.to(torch.complex128)
    thicknesses = thicknesses.unsqueeze(-1).repeat(1, 1, len(k))
    thicknesses = torch.permute(thicknesses, (0, 2, 1))
    
    # 计算反射率（TMM_solver返回(R, T)元组，我们只需要R）
    reflection, _ = TMM_solver(thicknesses, refractive_indices, k, params.theta, params.pol)

    return reflection


def _normalize_structure_entry(entry):
    """
    将任意表示形式的单个结构向量转换为统一格式。

    支持以下输入形式:
    1. 字典，包含键: lamda/lambda/lambda_um, n, k, h 或 layers
    2. (num_layers, 4) 的可迭代对象，列顺序为 [lamda, n, k, h]
    """
    if isinstance(entry, dict):
        lam_value = entry.get('lamda', entry.get('lambda', entry.get('lambda_um')))
        if lam_value is None:
            raise ValueError("结构字典缺少 'lamda'/'lambda'/'lambda_um' 字段")
        lam_value = float(lam_value)
        if 'layers' in entry:
            layers = np.asarray(entry['layers'], dtype=np.float64)
            if layers.ndim != 2 or layers.shape[1] != 4:
                raise ValueError("layers 字段必须是 (num_layers, 4) 形状的数组")
            n_values = layers[:, 1]
            k_values = layers[:, 2]
            h_values = layers[:, 3]
        else:
            try:
                n_values = np.asarray(entry['n'], dtype=np.float64)
                k_values = np.asarray(entry['k'], dtype=np.float64)
                h_values = np.asarray(entry['h'], dtype=np.float64)
            except KeyError as exc:
                raise ValueError("结构字典必须包含 'n', 'k', 'h' 键或 'layers' 键") from exc
        identifier = entry.get('id')
    else:
        layers = np.asarray(entry, dtype=np.float64)
        if layers.ndim != 2 or layers.shape[1] != 4:
            raise ValueError("结构数组必须是 (num_layers, 4) 形状且列顺序为 [lamda, n, k, h]")
        lam_column = layers[:, 0]
        lam_value = float(lam_column[0])
        if not np.allclose(lam_column, lam_value, atol=1e-9):
            raise ValueError("同一结构内部的所有层必须使用相同的波长值")
        n_values = layers[:, 1]
        k_values = layers[:, 2]
        h_values = layers[:, 3]
        identifier = None

    if not (n_values.size == k_values.size == h_values.size):
        raise ValueError("n, k, h 数据长度必须一致")

    return {
        'id': identifier,
        'lambda_um': lam_value,
        'n': n_values,
        'k': k_values,
        'h': h_values
    }

