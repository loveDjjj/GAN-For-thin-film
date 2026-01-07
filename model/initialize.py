import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch

from model.net import Generator, Discriminator


def initialize_models(params, device):
    """
    初始化 GAN 模型

    Args:
        params: 模型参数
        device: 运行设备(GPU/CPU)

    Returns:
        generator, discriminator: 初始化好的生成器和判别器模型
    """
    generator = Generator(params)

    wavelength_dim = len(params.k)
    discriminator = Discriminator(wavelength_dim)

    generator.to(device)
    discriminator.to(device)

    print("初始化模型: 生成器和判别器")

    return generator, discriminator
