#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""材料折射率数据库，统一从 data/Materials 下的 .mat 文件读取。
处理流程：读取 nm 波长数据 -> 转换为 µm -> 插值或组装 n/k."""
import os
import numpy as np
import h5py
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


MATERIALS_DIR = os.path.join(os.path.dirname(__file__), 'Materials')


def _resolve_mat_path(varname):
    """Resolve a material name to an actual .mat path under MATERIALS_DIR."""
    candidates = []
    if varname.lower().endswith('.mat'):
        candidates.append(varname)
    else:
        candidates.append(f"{varname}.mat")
    candidates.extend(os.listdir(MATERIALS_DIR))

    target_lower = varname.lower()
    for candidate in candidates:
        path = os.path.join(MATERIALS_DIR, candidate)
        if not os.path.isfile(path):
            continue
        if candidate.lower() == target_lower or candidate.lower().startswith(target_lower) or target_lower in candidate.lower():
            return path
    raise FileNotFoundError(f"未找到材料文件: {varname} (目录: {MATERIALS_DIR})")


def _load_mat(varname):
    """Load a .mat file, convert wavelength nm->um, and return wavelength + complex n."""
    path = _resolve_mat_path(varname)
    with h5py.File(path, 'r') as dataset:
        # 取首个数据集作为材料数据
        name = next(iter(dataset.items()))[0]
        array = dataset[name][:]

    # 输入格式: [wavelength(nm), n, k]
    wavelength_nm = array.T[:, 0]
    wavelength_um = wavelength_nm / 1000.0
    n_complex = array.T[:, 1] + 1j * array.T[:, 2]
    return wavelength_um, n_complex


# 材料索引函数 (mat 数据 -> µm 插值)
def index(varname):
    """
    加载指定材料的折射率数据文件并返回插值函数（输入/输出均以 µm 为单位的波长）
    """
    wavelength_um, n_complex = _load_mat(varname)
    n_interp = interp1d(
        wavelength_um.real,
        n_complex,
        kind='quadratic',
        bounds_error=False,
        fill_value="extrapolate",
    )
    return n_interp

class MatDatabase(object):
    """材料数据库类，基于 data/Materials 下的 .mat 数据（nm->µm 已转换）"""
    def __init__(self, material_key):
        super(MatDatabase, self).__init__()
        self.material_key = material_key
        self.num_materials = len(material_key)
        self.mat_database = self.build_database()

    def build_database(self):
        """构建材料数据库: {name: (wv_um, n_real, k_imag)}"""
        mat_database = {}
        for key in self.material_key:
            try:
                wv_um, n_complex = _load_mat(key)
                mat_database[key] = (wv_um, n_complex.real, n_complex.imag)
            except Exception as e:
                print(f'加载材料 {key} 时出错: {e}')
        return mat_database

    def interp_wv(self, wv_in, material_key, ignoreloss=False):
        """
        在指定波长(µm)处插值材料折射率数据
        Args:
            wv_in (tensor/array): 波长数组（µm）
            material_key (list): 材料名称列表
            ignoreloss (bool): 是否忽略损耗(虚部)
        Returns:
            材料折射率张量: 材料数 x 波长数
        """
        if hasattr(wv_in, 'device') and wv_in.device.type != 'cpu':
            wv_in_cpu = wv_in.cpu()
        else:
            wv_in_cpu = wv_in

        if hasattr(wv_in_cpu, 'numpy'):
            wv_in_np = wv_in_cpu.numpy()
        else:
            wv_in_np = wv_in_cpu

        n_data = np.zeros((len(material_key), wv_in_np.size))
        k_data = np.zeros((len(material_key), wv_in_np.size))

        for i in range(len(material_key)):
            mat = self.mat_database[material_key[i]]
            wv_um, n_real, k_imag = mat
            n_data[i, :] = np.interp(wv_in_np, wv_um, n_real)
            k_data[i, :] = np.interp(wv_in_np, wv_um, k_imag)

        if ignoreloss:
            return torch.tensor(n_data)
        else:
            return torch.tensor(n_data), torch.tensor(k_data)
