"""工具函数和配置加载模块"""
import os
import json
import logging
import yaml
import torch
import numpy as np


class Params():
    """加载和管理参数的类

    示例:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # 修改参数值
    ```
    """

    def __init__(self, json_path=None):
        if json_path is not None:
            self.update(json_path)

    def save(self, json_path):
        """将参数保存到JSON文件"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """从JSON文件加载参数"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """提供字典形式访问Params实例的方法，如：params.dict['learning_rate']"""
        return self.__dict__


def load_config(config_path):
    """加载YAML配置文件
    
    Args:
        config_path: YAML配置文件的路径
        
    Returns:
        包含配置的字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config


def update_params(params, config):
    """从配置字典更新参数对象
    
    Args:
        params: 参数对象
        config: 配置字典
    
    Returns:
        更新后的参数对象
    """
    # 结构参数
    if 'structure' in config:
        if 'N_layers' in config['structure']:
            params.N_layers = config['structure']['N_layers']
        if 'pol' in config['structure']:
            params.pol = config['structure']['pol']
        if 'thickness_sup' in config['structure']:
            params.thickness_sup = config['structure']['thickness_sup']
        if 'thickness_bot' in config['structure']:
            params.thickness_bot = config['structure']['thickness_bot']
    
    # 材料参数
    if 'materials' in config and 'materials_list' in config['materials']:
        params.materials = config['materials']['materials_list']
    
    # 光学参数
    if 'optics' in config:
        if 'wavelength_range' in config['optics']:
            params.wavelength_range = config['optics']['wavelength_range']
        if 'samples_total' in config['optics']:
            params.samples_total = config['optics']['samples_total']
        if 'theta' in config['optics']:
            params.theta = config['optics']['theta']
        if 'n_top' in config['optics']:
            params.n_top = config['optics']['n_top']
        if 'n_bot' in config['optics']:
            params.n_bot = config['optics']['n_bot']
        if 'lorentz_width' in config['optics']:
            params.lorentz_width = config['optics']['lorentz_width']
        if 'lorentz_center_range' in config['optics']:
            params.lorentz_center_range = config['optics']['lorentz_center_range']
    
    # 生成器参数
    if 'generator' in config:
        if 'thickness_noise_dim' in config['generator']:
            params.thickness_noise_dim = config['generator']['thickness_noise_dim']
        if 'material_noise_dim' in config['generator']:
            params.material_noise_dim = config['generator']['material_noise_dim']
        if 'alpha_sup' in config['generator']:
            params.alpha_sup = config['generator']['alpha_sup']
        if 'alpha' in config['generator']:
            params.alpha = config['generator']['alpha']
    
    # 训练参数
    if 'training' in config:
        if 'epochs' in config['training']:
            params.epochs = config['training']['epochs']
        if 'batch_size' in config['training']:
            params.batch_size = config['training']['batch_size']
        if 'save_interval' in config['training']:
            params.save_interval = config['training']['save_interval']
        if 'd_steps' in config['training']:
            params.d_steps = config['training']['d_steps']
        if 'g_steps' in config['training']:
            params.g_steps = config['training']['g_steps']
    
    # 优化器参数
    if 'optimizer' in config:
        if 'lr_gen' in config['optimizer']:
            params.lr_gen = config['optimizer']['lr_gen']
        if 'lr_disc' in config['optimizer']:
            params.lr_disc = config['optimizer']['lr_disc']
        if 'beta1' in config['optimizer']:
            params.beta1 = config['optimizer']['beta1']
        if 'beta2' in config['optimizer']:
            params.beta2 = config['optimizer']['beta2']
    
    # 设置标志位
    params.user_define = False
    
    return params

# 设置日志记录器
def set_logger(log_path):
    """设置日志记录器，将输出保存到终端和文件

    Args:
        log_path: 日志文件路径
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 日志输出到文件
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # 日志输出到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def generate_wavelength_samples(config):
    """根据配置生成波长采样点

    支持两种采样模式：
    1. uniform: 均匀采样
    2. segmented: 分段采样（边缘稀疏 + 中心密集）

    Args:
        config: 包含 optics 配置的字典

    Returns:
        wavelengths: 波长张量 (torch.Tensor)
        k: 波数张量 (torch.Tensor)
    """
    optics = config.get('optics', {})
    sampling_mode = optics.get('sampling_mode', 'uniform')

    if sampling_mode == 'segmented':
        # 分段采样模式
        segments = optics.get('segments', {})

        if not segments:
            raise ValueError("分段采样模式需要配置 'segments' 参数")

        wavelength_list = []

        # 按顺序处理各段：left, center, right
        for seg_name in ['left', 'center', 'right']:
            if seg_name not in segments:
                continue
            seg = segments[seg_name]
            seg_range = seg.get('range', [])
            seg_samples = seg.get('samples', 0)

            if len(seg_range) != 2 or seg_samples <= 0:
                raise ValueError(f"分段 '{seg_name}' 配置无效: range={seg_range}, samples={seg_samples}")

            # 生成该段的波长采样点
            # 注意：为避免边界重复，除第一段外都排除起始点
            if seg_name == 'left':
                seg_wavelengths = torch.linspace(seg_range[0], seg_range[1], seg_samples)
            else:
                # 排除起始点，避免与前一段的终点重复
                seg_wavelengths = torch.linspace(seg_range[0], seg_range[1], seg_samples + 1)[1:]

            wavelength_list.append(seg_wavelengths)

        wavelengths = torch.cat(wavelength_list)

        # 打印采样信息
        total_samples = len(wavelengths)
        print(f"分段采样模式: 总采样点数 = {total_samples}")
        for seg_name in ['left', 'center', 'right']:
            if seg_name in segments:
                seg = segments[seg_name]
                seg_range = seg['range']
                seg_samples = seg['samples']
                density = seg_samples / (seg_range[1] - seg_range[0])
                print(f"  {seg_name}: {seg_range[0]:.2f}-{seg_range[1]:.2f} μm, "
                      f"{seg_samples} 点, 密度 = {density:.1f} 点/μm")
    else:
        # 均匀采样模式
        wavelength_range = optics.get('wavelength_range', [3, 5.5])
        samples_total = optics.get('samples_total', 5000)

        wavelengths = torch.linspace(wavelength_range[0], wavelength_range[1], samples_total)

        density = samples_total / (wavelength_range[1] - wavelength_range[0])
        print(f"均匀采样模式: {wavelength_range[0]:.2f}-{wavelength_range[1]:.2f} μm, "
              f"{samples_total} 点, 密度 = {density:.1f} 点/μm")

    # 计算波数 k = 2π/λ
    k = 2 * np.pi / wavelengths

    return wavelengths, k
