import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差连接块，用于增强信息流动和梯度传播"""
    def __init__(self, in_features, hidden_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # 获取参数或使用默认值
        self.thickness_noise_dim = getattr(params, 'thickness_noise_dim', 32)
        self.material_noise_dim = getattr(params, 'material_noise_dim', 32)
        self.hidden_dim = getattr(params, 'generator_hidden_dim', 128)  # 新增隐藏层维度
        
        self.thickness_bot = params.thickness_bot
        self.thickness_sup = params.thickness_sup
        self.N_layers = params.N_layers
        self.M_materials = params.M_materials
        
        # 添加折射率实部和虚部
        self.n_database = params.n_database.view(1, 1, params.M_materials, -1).cuda() # 实部
        self.k_database = params.k_database.view(1, 1, params.M_materials, -1).cuda() # 虚部
        
        # 增强的厚度生成网络 - 多层MLP + 残差连接
        self.thickness_encoder = nn.Sequential(
            nn.Linear(self.thickness_noise_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        self.thickness_res_block = ResidualBlock(self.hidden_dim, self.hidden_dim // 2)
        
        self.thickness_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim // 2, self.N_layers)
        )
        
        # 增强的材料选择网络 - 多层MLP + 残差连接
        self.material_encoder = nn.Sequential(
            nn.Linear(self.material_noise_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        self.material_res_block = ResidualBlock(self.hidden_dim, self.hidden_dim // 2)
        
        self.material_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim * 2, self.N_layers * self.M_materials)
        )
        
        # 层间关系模块 - 可选的注意力机制来建模层间的依赖关系
        self.layer_interaction = nn.Sequential(
            nn.Linear(self.N_layers * (1 + self.M_materials), self.N_layers * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.N_layers * 4, self.N_layers * (1 + self.M_materials))
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重，使用Kaiming初始化改善训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, thickness_noise, material_noise, alpha):
        batch_size = thickness_noise.size(0)

        def _sanitize(x, clamp_val=10.0):
            x = torch.nan_to_num(x, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
            return torch.clamp(x, -clamp_val, clamp_val)
        
        # 厚度生成路径
        thickness_features = self.thickness_encoder(thickness_noise)
        thickness_features = self.thickness_res_block(thickness_features)
        thickness_logits = self.thickness_decoder(thickness_features)
        thickness_logits = _sanitize(thickness_logits)
        
        # 材料选择路径
        material_features = self.material_encoder(material_noise)
        material_features = self.material_res_block(material_features)
        material_logits = self.material_decoder(material_features)
        material_logits = _sanitize(material_logits)
        material_logits = material_logits.view(batch_size, self.N_layers, self.M_materials)
        
        # 生成厚度和材料选择概率
        thicknesses_raw = thickness_logits  # [batch_size, N_layers]
        material_probs_raw = F.softmax(material_logits * alpha, dim=2)  # [batch_size, N_layers, M_materials]
        
        # 可选：层间交互建模 - 厚度和材料选择之间的关系处理
        if hasattr(self, 'layer_interaction'):
            # 将厚度和材料信息合并
            combined = torch.cat([
                thicknesses_raw.unsqueeze(2),  # [batch_size, N_layers, 1]
                material_probs_raw            # [batch_size, N_layers, M_materials]
            ], dim=2)
            
            # 展平进行处理
            flat_combined = combined.view(batch_size, -1)
            flat_refined = self.layer_interaction(flat_combined)
            refined = flat_refined.view(batch_size, self.N_layers, 1 + self.M_materials)
            
            # 分离厚度和材料信息
            thicknesses_refined = _sanitize(refined[:, :, 0])
            material_probs_refined = _sanitize(refined[:, :, 1:])
            
            # 确保材料概率满足条件（归一化）
            material_probs = F.softmax(material_probs_refined * alpha, dim=2)
        else:
            thicknesses_refined = thicknesses_raw
            material_probs = material_probs_raw
            
        # 应用物理约束 - 厚度范围限制
        thicknesses = self.thickness_bot + torch.sigmoid(thicknesses_refined) * (self.thickness_sup - self.thickness_bot)
        thicknesses = torch.nan_to_num(
            thicknesses,
            nan=(self.thickness_bot + self.thickness_sup) / 2.0,
            posinf=self.thickness_sup,
            neginf=self.thickness_bot,
        )
        thicknesses = torch.clamp(thicknesses, self.thickness_bot, self.thickness_sup)
        
        # 准备材料概率和复折射率计算
        material_probs = torch.nan_to_num(
            material_probs,
            nan=1.0 / max(self.M_materials, 1),
            posinf=1.0 / max(self.M_materials, 1),
            neginf=0.0,
        )
        material_probs = material_probs / material_probs.sum(dim=2, keepdim=True).clamp_min(1e-6)
        P = material_probs.unsqueeze(-1)  # [batch_size, N_layers, M_materials, 1]

        # 计算完整的复折射率 - 根据材料选择概率的加权和
        n_part = torch.sum(P * self.n_database, dim=2)  # 实部 [batch_size, N_layers, wavelengths]
        k_part = torch.sum(P * self.k_database, dim=2)  # 虚部 [batch_size, N_layers, wavelengths]
        refractive_indices = n_part + 1j * k_part  # 复折射率
        
        return (thicknesses, refractive_indices, P.squeeze())

    # 向后兼容的方法，用于支持旧代码调用方式
    def legacy_forward(self, noise, alpha):
        # 拆分噪声向量
        split_point = noise.size(1) // 2
        thickness_noise = noise[:, :split_point]
        material_noise = noise[:, split_point:]
        return self.forward(thickness_noise, material_noise, alpha)


class Discriminator(nn.Module):
    """判别器网络：判断吸收率曲线是真实的洛伦兹曲线还是生成的曲线"""
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        # 吸收率曲线的输入维度(波长点数)
        self.input_dim = input_dim
        
        # 使用1D卷积捕捉吸收率曲线的特征 - 简化网络结构
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层卷积
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第三层卷积
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 第四层卷积
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 计算卷积后的特征维度
        conv_output_dim = self.calculate_conv_output_dim()
        
        # 全连接层 - 简化结构
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)  # 无sigmoid激活，直接输出线性值
        )
        
        # 更保守的初始化方法
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用更保守的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def calculate_conv_output_dim(self):
        """计算卷积网络输出的特征维度"""
        dim = self.input_dim
        # 每层卷积的输出维度变化: dimension = (dimension - kernel_size + 2*padding) / stride + 1
        # 第一层卷积后的维度
        dim = (dim - 5 + 2*2) // 2 + 1
        # 第二层卷积后的维度
        dim = (dim - 5 + 2*2) // 2 + 1
        # 第三层卷积后的维度
        dim = (dim - 5 + 2*2) // 2 + 1
        # 第四层卷积后的维度
        dim = (dim - 5 + 2*2) // 2 + 1
        # 最终卷积输出的特征数 = 通道数 * 维度
        return 256 * dim
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 吸收率曲线，形状为 [batch_size, sequence_length]
        """
        # 添加通道维度 [batch_size, 1, sequence_length]
        x = x.unsqueeze(1)
        # 通过卷积层
        x = self.conv_layers(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 通过全连接层 - 无sigmoid，直接返回线性输出
        return self.fc_layers(x)


class SelfAttention(nn.Module):
    """自注意力机制，用于捕捉谱线特征间的关系"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # 查询、键、值变换
        self.query = nn.Conv1d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # 注意力比例因子
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # 输入: [batch_size, channels, sequence_length]
        batch_size, C, width = x.size()
        
        # 计算查询、键、值
        proj_query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)  # [B, W, C//8]
        proj_key = self.key(x).view(batch_size, -1, width)  # [B, C//8, W]
        energy = torch.bmm(proj_query, proj_key)  # [B, W, W]
        attention = F.softmax(energy, dim=-1)  # [B, W, W]
        
        proj_value = self.value(x).view(batch_size, -1, width)  # [B, C, W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, W]
        out = out.view(batch_size, C, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


