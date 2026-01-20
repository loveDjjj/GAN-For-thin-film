"""工具函数和配置加载模块"""
import os
import json
import logging
import yaml


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
