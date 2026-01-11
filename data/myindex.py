#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""材料折射率数据库，统一从 data/Materials 下的 .mat 文件读取。
处理流程：读取 nm 波长数据 -> 转换为 µm -> 插值或组装 n/k."""
import os
import numpy as np
import h5py
import torch
from scipy import io as sio
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


MATERIALS_DIR = os.path.join(os.path.dirname(__file__), 'Materials')


def _resolve_mat_paths(varname):
    """Return an ordered list of candidate .mat paths under MATERIALS_DIR."""
    target_lower = varname.lower()

    exact_candidates = [varname]
    if not target_lower.endswith('.mat'):
        exact_candidates.append(f"{varname}.mat")

    files = sorted([f for f in os.listdir(MATERIALS_DIR) if os.path.isfile(os.path.join(MATERIALS_DIR, f))])

    ordered = []

    # 精确匹配
    for cand in exact_candidates:
        for f in files:
            if f.lower() == cand.lower():
                ordered.append(os.path.join(MATERIALS_DIR, f))

    # 前缀匹配
    for f in files:
        if f.lower().startswith(target_lower):
            ordered.append(os.path.join(MATERIALS_DIR, f))

    # 子串匹配
    for f in files:
        if target_lower in f.lower():
            ordered.append(os.path.join(MATERIALS_DIR, f))

    # 去重同时保留顺序
    seen = set()
    unique_ordered = []
    for p in ordered:
        if p not in seen:
            unique_ordered.append(p)
            seen.add(p)

    if not unique_ordered:
        raise FileNotFoundError(f"未找到材料文件: {varname} (目录: {MATERIALS_DIR})")
    return unique_ordered


def _load_mat(varname):
    """Load a .mat file, convert wavelength nm->um, and return wavelength + complex n."""
    candidate_paths = _resolve_mat_paths(varname)

    def _from_array(array):
        # 接受形状 (N,3) 或 (3,N)
        arr = np.array(array)
        if arr.ndim != 2:
            raise ValueError(f"材料文件 {path} 中的数据不是二维数组，shape={arr.shape}")
        if arr.shape[1] == 3:
            arr3 = arr
        elif arr.shape[0] == 3:
            arr3 = arr.T
        else:
            raise ValueError(f"材料文件 {path} 中的数据无法解析，shape={arr.shape}")
        wavelength_nm = arr3[:, 0]
        wavelength_um = wavelength_nm / 1000.0
        n_complex = arr3[:, 1] + 1j * arr3[:, 2]
        return wavelength_um, n_complex

    errors = []
    for path in candidate_paths:
        # 优先使用 HDF5 读取
        try:
            with h5py.File(path, 'r') as dataset:
                name = next(iter(dataset.items()))[0]
                array = dataset[name][:]
            return _from_array(array.T if array.shape[0] == 3 else array)
        except Exception as e_h5:
            errors.append(f"HDF5读取失败({path}): {e_h5}")

        # 尝试使用 scipy.io 读取 (Matlab v5/v6)
        try:
            mat = sio.loadmat(path)
            for key, value in mat.items():
                if key.startswith('__'):
                    continue
                if isinstance(value, np.ndarray) and value.size > 0:
                    return _from_array(value)
            errors.append(f"未找到有效数组({path})")
        except Exception as e_mat:
            errors.append(f"scipy读取失败({path}): {e_mat}")

    raise RuntimeError("读取材料文件失败: " + "; ".join(errors))


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
            if material_key[i] not in self.mat_database:
                raise KeyError(f"材料 {material_key[i]} 未成功加载，无法插值。检查文件是否存在且格式正确。")
            mat = self.mat_database[material_key[i]]
            wv_um, n_real, k_imag = mat
            n_data[i, :] = np.interp(wv_in_np, wv_um, n_real)
            k_data[i, :] = np.interp(wv_in_np, wv_um, k_imag)

        if ignoreloss:
            return torch.tensor(n_data)
        else:
            return torch.tensor(n_data), torch.tensor(k_data)
