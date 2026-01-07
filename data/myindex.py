#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
材料数据库模块
包含材料折射率数据加载和处理功能
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 材料索引函数(来自原始myindex.py)
def index(varname):
    """
    加载指定材料的折射率数据文件
    
    Args:
        varname: 材料文件名称(.mat格式)
    
    Returns:
        材料折射率的插值函数
    """
    database_path = os.path.join(os.path.dirname(__file__), 'Materials')
    
    dataset = h5py.File(os.path.join(database_path, varname))
    
    variables = dataset.items()
    for var in variables:
        name = var[0]
    
    array = dataset[name][:]
    
    n = np.zeros((np.shape(array)[1], 2), dtype=complex)
    # 输入折射率应该遵循格式 - 波长/nm n k
    
    n[:, 0] = array.T[:, 0]
    n[:, 1] = array.T[:, 1] + 1j * array.T[:, 2]
    
    n_interp = interp1d(n[:, 0].real, n[:, 1], kind='quadratic')
    
    return n_interp

def Materials_dir():
    """
    获取材料目录列表
    
    Returns:
        可用材料列表
    """
    database_path = os.path.join(os.path.dirname(__file__), 'Materials')
    
    Materials_list = os.listdir(database_path)
    
    return Materials_list

def nk(varname):
    """
    显示材料的折射率数据并绘图
    
    Args:
        varname: 材料文件名称(.mat格式)
    
    Returns:
        材料的折射率数据数组
    """
    database_path = os.path.join(os.path.dirname(__file__), 'Materials')
    
    dataset = h5py.File(os.path.join(database_path, varname))
    
    print(dataset)
    
    variables = dataset.items()
    
    for var in variables:
        name = var[0]
        print("Name ", name)
    
    array = dataset[name][:]
    plt.plot(array.T[:, 0], array.T[:, 1], 'Purple', array.T[:, 0], array.T[:, 2], 'Blue')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Refractive index')
    plt.legend(['n', 'k'])
    plt.title(varname)
    
    return array

# 材料数据库类(来自原始material_database.py)
class MatDatabase(object):
    """材料数据库类
    参数: 
        material_key: 材料名称列表
    """
    def __init__(self, material_key):
        super(MatDatabase, self).__init__()
        self.material_key = material_key
        self.num_materials = len(material_key)
        self.mat_database = self.build_database()

    def build_database(self):
        """
        构建材料数据库
        
        Returns:
            包含材料折射率数据的字典
        """
        mat_database = {}
        
        # 读取每种材料的色散数据
        for i in range(self.num_materials):
            file_name = os.path.join(os.path.dirname(__file__), 'material_data', f'mat_{self.material_key[i]}.xlsx')
            
            try: 
                A = np.array(pd.read_excel(file_name))
                mat_database[self.material_key[i]] = (A[:, 0], A[:, 1], A[:, 2])
            except Exception as e:
                print(f'加载材料 {self.material_key[i]} 时出错: {e}')

        return mat_database

    def interp_wv(self, wv_in, material_key, ignoreloss=False):
        """
        在指定波长处插值材料折射率数据
        
        Args:
            wv_in (tensor): 波长数组
            material_key (list): 材料名称列表
            ignoreloss (bool): 是否忽略损耗(虚部)
            
        Returns:
            材料折射率张量: 材料数 x 波长数
        """
        # 确保wv_in在CPU上
        if hasattr(wv_in, 'device') and wv_in.device.type != 'cpu':
            wv_in_cpu = wv_in.cpu()
        else:
            wv_in_cpu = wv_in
            
        # 转换为NumPy数组以便使用np.interp
        if hasattr(wv_in_cpu, 'numpy'):
            wv_in_np = wv_in_cpu.numpy()
        else:
            wv_in_np = wv_in_cpu
            
        n_data = np.zeros((len(material_key), wv_in_np.size))
        k_data = np.zeros((len(material_key), wv_in_np.size))
        
        for i in range(len(material_key)):
            mat = self.mat_database[material_key[i]]
            n_data[i, :] = np.interp(wv_in_np, mat[0], mat[1])
            k_data[i, :] = np.interp(wv_in_np, mat[0], mat[2])

        if ignoreloss:
            return torch.tensor(n_data)
        else:
            return torch.tensor(n_data), torch.tensor(k_data) 