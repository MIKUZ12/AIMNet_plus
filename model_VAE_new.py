import torch
# from utils.expert import weight_sum_var, ivw_aggregate_var
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.functional import softplus

def gaussian_reparameterization_var(means, var, times=1):
    # 检查并处理输入
    if torch.isnan(means).any():
        print("警告: means中包含NaN")
        means = torch.nan_to_num(means, nan=0.0)
    
    if torch.isnan(var).any():
        print("警告: var中包含NaN")
        var = torch.nan_to_num(var, nan=1e-6)
    
    # 确保方差为正值
    var = torch.clamp(var, min=1e-6)
    
    # 计算标准差
    std = torch.sqrt(var)
    
    # 重参数化技巧
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    
    # 最终检查
    if torch.isnan(res).any():
        print("警告: 重参数化结果中包含NaN")
        res = torch.nan_to_num(res, nan=0.0)
    
    return res/times

def manual_gaussian_log_prob_stable(x, mu, var, eps=1e-8):
    """更稳定的手动高斯分布对数概率计算
    
    Args:
        x: 样本点 [batch_size, dim]
        mu: 均值 [batch_size, dim]
        var: 方差 [batch_size, dim]
        eps: 数值稳定性常数
    Returns:
        log_prob: 对数概率 [batch_size]
    """
    # 检查和处理输入以避免NaN
    if torch.isnan(x).any():
        print("警告: x中包含NaN")
        x = torch.nan_to_num(x, nan=0.0)
    
    if torch.isnan(mu).any():
        print("警告: mu中包含NaN")
        mu = torch.nan_to_num(mu, nan=0.0)
    
    if torch.isnan(var).any():
        print("警告: var中包含NaN")
        var = torch.nan_to_num(var, nan=eps)
    
    # 确保方差为正值
    var = torch.clamp(var, min=eps)
    
    # 计算常数项 -0.5 * log(2π)
    const = -0.5 * np.log(2 * np.pi)
    
    # 计算 -0.5 * log(σ²)，添加数值稳定性
    log_det = -0.5 * torch.log(var + eps)
    
    # 计算 -0.5 * (x - μ)²/σ²，使用更稳定的计算方式
    diff = x - mu
    # 限制diff的大小，防止极端值
    diff = torch.clamp(diff, min=-1e6, max=1e6)
    # 避免除以接近0的数
    scaled_diff = diff / torch.sqrt(var + eps)
    mahalanobis = -0.5 * scaled_diff.pow(2)
    
    # 每个维度的对数概率
    log_prob_per_dim = const + log_det + mahalanobis
    
    # 检查是否有无穷值或NaN
    if torch.isnan(log_prob_per_dim).any() or torch.isinf(log_prob_per_dim).any():
        print("警告: 对数概率计算中出现NaN或Inf")
        log_prob_per_dim = torch.nan_to_num(log_prob_per_dim, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 在求和前检查数值范围，避免极端值
    log_prob_per_dim = torch.clamp(log_prob_per_dim, min=-100, max=100)
    
    # 独立多维高斯分布的对数概率是各维度对数概率之和
    log_prob = torch.sum(log_prob_per_dim, dim=-1)
    
    # 最终检查
    if torch.isnan(log_prob).any():
        print("警告: 最终结果中包含NaN")
        log_prob = torch.zeros_like(log_prob)
    
    return log_prob

def manual_multivariate_gaussian_log_prob(x, mu, cov, eps=1e-6):
    """手动计算多元高斯分布的对数概率
    Args:
        x: 样本点 [batch_size, dim]
        mu: 均值 [batch_size, dim]
        cov: 协方差矩阵 [batch_size, dim, dim] 或对角线方差 [batch_size, dim]
        eps: 数值稳定性常数
    Returns:
        log_prob: 对数概率 [batch_size]
    """
    batch_size, dim = x.shape
    
    # 如果cov是对角线方差，转换为对角矩阵
    if cov.dim() == 2:
        cov = torch.diag_embed(torch.clamp(cov, min=eps))
    
    # 计算常数项 -0.5 * dim * log(2π)
    const = -0.5 * dim * torch.log(torch.tensor(2 * np.pi))
    
    # 计算行列式的对数 -0.5 * log(det(Σ))
    # 对于对角矩阵，行列式就是对角线元素的乘积
    log_det = -0.5 * torch.log(torch.diagonal(cov, dim1=-2, dim2=-1).prod(dim=-1) + eps)
    
    # 计算马氏距离 -0.5 * (x - μ)ᵀ Σ⁻¹ (x - μ)
    # 对于对角矩阵，逆矩阵就是对角线元素的倒数
    diff = x - mu
    inv_cov = torch.inverse(cov + eps * torch.eye(dim, device=cov.device).unsqueeze(0))
    mahalanobis = -0.5 * torch.bmm(
        torch.bmm(diff.unsqueeze(1), inv_cov),
        diff.unsqueeze(-1)
    ).squeeze()
    
    log_prob = const + log_det + mahalanobis
    return log_prob

def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)/(label.sum(dim=1,keepdim=True)+1e-8)
    new_x =  x_embedding*inc_V_ind.T.unsqueeze(-1) + fea.unsqueeze(0)*(1-inc_V_ind.T.unsqueeze(-1))
    return new_x
class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,dropout_rate=0.,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            # layers.append(nn.Dropout(dropout_rate))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                # layers.append(nn.Dropout(dropout_rate))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            # layers.append(nn.Dropout(dropout_rate))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
            # x = x + y
        return x

class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
            nn.Tanh(),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg_idx = torch.randperm(x1.size(0))
        neg = self.net(torch.cat([x1[neg_idx], x2], 1))
        # neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean()

class sharedQz_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(sharedQz_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        hidden_features = self.mlp(x)
        z_mu = self.z_loc(hidden_features)
        z_sca_raw = self.z_sca(hidden_features)
        
        # 额外的安全检查
        z_sca = torch.clamp(z_sca_raw, min=1e-6)
        
        return z_mu, z_sca
    
class inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        hidden_features = self.mlp(x)
        # class_feature  = self.z
        return hidden_features
    
class Px_generation_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[512]):
        super(Px_generation_mlp, self).__init__()
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim,final_act=False,final_norm=False)
        # self.transfer_act = nn.ReLU
        # self.px_layer = mlp_layers_creation(self.z_dim, self.x_dim, self.layers, self.transfer_act)
        # self.px_z = nn.Sequential(*self.px_layer)
    def forward(self, z):
        xr = self.mlp(z)
        return xr

class VAE(nn.Module):
    def __init__(self, d_list,z_dim,class_num):
        super(VAE, self).__init__()
        self.x_dim_list = d_list
        self.k = class_num
        self.z_dim = z_dim
        self.num_views = len(d_list)
    # self.switch_layers = switch_layers(z_dim,self.num_views)
        self.z_inference_s = []
        self.z_inference_p = []

        self.mu2 = nn.Parameter(torch.full((self.z_dim,),1.), requires_grad=False)
        self.sigma = nn.Parameter(torch.full((self.z_dim,),2.), requires_grad=False)
        self.prior2 = torch.distributions.Normal(loc=self.mu2, scale=self.sigma)
        self.prior2 = torch.distributions.Independent(self.prior2, 1)
        # 为每一个视图都创建一个独立的编码器
        for v in range(self.num_views):
            self.z_inference_s.append(inference_mlp(self.x_dim_list[v], self.z_dim))
            self.z_inference_p.append(inference_mlp(self.x_dim_list[v], self.z_dim))
        self.qz_inference_s = nn.ModuleList(self.z_inference_s)
        self.qz_inference_p = nn.ModuleList(self.z_inference_p)
        self.qz_inference = nn.ModuleList([inference_mlp(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)])
        # 定义共享潜在分布的推理模块,通过 MLP 提取特征，再通过线性层分别输出潜在分布的均值z_loc和方差z_sca(使用 Softplus 确保正性)
        self.qz_inference_header = sharedQz_inference_mlp(self.z_dim, self.z_dim)
        self.x_generation_s = []
        self.x_generation_p = []
        for v in range(self.num_views):
            # 为每一个视图都创建一个独立的解码器
            self.x_generation_s.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
            self.x_generation_p.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
        self.px_generation_s = nn.ModuleList(self.x_generation_s)
        self.px_generation_p = nn.ModuleList(self.x_generation_p)


        self.mi_estimator = []
        for v in range(self.num_views):
            self.mi_estimator.append(MIEstimator(self.z_dim, self.z_dim))
        self.mi_estimator = nn.ModuleList(self.mi_estimator)

    def inference_z(self, x_list):
        uniview_mu_list = []
        uniview_sca_list = []
        uniview_mu_s_list = []
        uniview_sca_s_list = []
        uniview_mu_p_list = []
        uniview_sca_p_list = []
        
        for v in range(self.num_views):
            # 正确检查0值
            zeros_mask = (x_list[v] == 0)
            zero_count = zeros_mask.sum().item()
            total_elements = x_list[v].numel()
            
            if zero_count > 0:
                print(f"视图 {v} 中有 {zero_count} 个零值 (占比 {zero_count/total_elements*100:.2f}%)")
                
                # 选项1: 为所有0值添加小噪声
                noise = torch.randn_like(x_list[v]) * 1e-4
                # 只有在0值的位置添加噪声
                x_list[v] = torch.where(zeros_mask, noise, x_list[v])
                
                # 选项2: 用小的非零值替换所有0值
                # small_value = 1e-6
                # x_list[v] = torch.where(zeros_mask, torch.ones_like(x_list[v]) * small_value, x_list[v])
            
            # 检查输入是否有NaN
            if torch.isnan(x_list[v]).any():
                print(f"视图 {v} 输入中包含NaN")
                # 替换NaN为0
                x_list[v] = torch.nan_to_num(x_list[v], nan=1e-6)
            
            # 以下是原有的特征提取代码
            fea = self.qz_inference[v](x_list[v])
            fea_s = self.qz_inference_s[v](x_list[v])
            fea_p = self.qz_inference_p[v](x_list[v])
            
            # 检查中间特征是否有NaN
            if torch.isnan(fea).any():
                print(f"视图 {v} 特征中包含NaN")
                fea = torch.nan_to_num(fea, nan=0.0)
            
            # 通过共享头生成每个特征的潜在分布均值和方差
            z_mu_v, z_sca_v = self.qz_inference_header(fea)
            z_mu_v_s, z_sca_v_s = self.qz_inference_header(fea_s)
            z_mu_v_p, z_sca_v_p = self.qz_inference_header(fea_p)
            
            # 检查并处理方差
            z_sca_v = torch.clamp(z_sca_v, min=1e-6)  # 确保方差为正值
            z_sca_v_s = torch.clamp(z_sca_v_s, min=1e-6)
            z_sca_v_p = torch.clamp(z_sca_v_p, min=1e-6)
            
            # 检查并处理均值
            if torch.isnan(z_mu_v).any():
                print(f"视图 {v} 均值中包含NaN")
                z_mu_v = torch.nan_to_num(z_mu_v, nan=0.0)
            
            # 添加到列表
            uniview_mu_list.append(z_mu_v)
            uniview_sca_list.append(z_sca_v)
            uniview_mu_s_list.append(z_mu_v_s)
            uniview_sca_s_list.append(z_sca_v_s)
            uniview_mu_p_list.append(z_mu_v_p)
            uniview_sca_p_list.append(z_sca_v_p)
        
        return uniview_mu_list, uniview_sca_list, uniview_mu_s_list, uniview_sca_s_list, uniview_mu_p_list, uniview_sca_p_list
    
    
    def poe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        # mask_matrix = torch.stack(mask, dim=0)
        mask_matrix_new = torch.cat([torch.ones([1,mask_matrix.shape[1],mask_matrix.shape[2]]).cuda(),mask_matrix],dim=0)
        p_z_mu = torch.zeros([1,mu.shape[1],mu.shape[2]]).cuda()
        p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        mu_new = torch.cat([p_z_mu,mu],dim=0)
        var_new = torch.cat([p_z_var,var],dim=0)
        exist_mu = mu_new * mask_matrix_new
        T = 1. / (var_new+eps)
        if torch.sum(torch.isnan(exist_mu)).item()>0:
            print('.')
        if torch.sum(torch.isinf(T)).item()>0:
            print('.')
        exist_T = T * mask_matrix_new
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)
        # if torch.sum(torch.isnan(aggregate_var)).item()>0:
        #     print('.')
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / (aggregate_T + eps)
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        return aggregate_mu, aggregate_var
    
    def weighted_poe_aggregate(self, mu, var, weights=None, mask=None, eps=1e-5):
        """
        按照加权均值和加权方差公式进行融合，直接生成正确形状的输出
        
        参数:
        - mu: 各视图的均值列表，每个元素形状为 [N, D]
        - var: 各视图的方差列表，每个元素形状为 [N, D]
        - weights: 各视图的权重，形状为 [V]
        - mask: 视图掩码 [N, V]
        - eps: 数值稳定性常数
        
        返回:
        - aggregate_mu: 融合后的均值，形状为 [N, D]
        - aggregate_var: 融合后的方差，形状为 [N, D]
        """
        
        # 处理权重
        if weights is None:
            # 如果没有提供权重，使用均匀权重
            weights = torch.ones(mu.shape[0], device=mu.device)  # [V]
        
        # 将权重改变形状以便进行广播
        weights = weights.view(-1, 1, 1)  # [V, 1, 1]
        weights = weights.expand(-1, mu.shape[1], 1)  # [V, N, 1]
        weights = weights.to(mu.device)
        # 计算权重总和
        weights_sum = torch.sum(weights, dim=0) + eps  # [1, N, 1]
        
        # 1. 计算融合均值：μ_fused = Σ(w_i * μ_i) / Σ(w_i)
        weighted_mu = mu * weights  # [V, N, D]
        aggregate_mu = torch.sum(weighted_mu, dim=0) / weights_sum  # [N, D]
        
        # 2. 计算融合方差：σ²_fused = Σ(w_i * (σ²_i + μ²_i)) / Σ(w_i) - μ²_fused
        mu_squared = mu ** 2  # [V, N, D]
        weighted_var_plus_mu_squared = weights * (var + mu_squared)  # [V, N, D]
        
        # 累加所有视图的加权方差和均值平方
        sum_weighted_var_plus_mu_squared = torch.sum(weighted_var_plus_mu_squared, dim=0)  # [N, D]
        
        # 计算归一化后的总和
        normalized_sum = sum_weighted_var_plus_mu_squared / weights_sum.squeeze(0)  # [N, D]
        
        # 减去融合均值的平方
        aggregate_var = normalized_sum - aggregate_mu ** 2  # [N, D]
        
        # 确保方差为正
        aggregate_var = torch.clamp(aggregate_var, min=eps)  # [N, D]
        
        # 数值检查
        if torch.isnan(aggregate_mu).any():
            print('警告: 融合均值中存在NaN')
            aggregate_mu = torch.nan_to_num(aggregate_mu, nan=0.0)
        
        if torch.isnan(aggregate_var).any():
            print('警告: 融合方差中存在NaN')
            aggregate_var = torch.nan_to_num(aggregate_var, nan=eps)
        
        if torch.isinf(aggregate_var).any():
            print('警告: 融合方差中存在Inf')
            aggregate_var = torch.clamp(aggregate_var, min=eps, max=1e6)
        
        # 直接返回 [N, D] 形状的结果，不需要 squeeze
        return aggregate_mu, aggregate_var

    def moe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        exist_mu = mu * mask_matrix
        exist_var = var * mask_matrix
        aggregate_var = exist_var.sum(dim=0)
        aggregate_mu = exist_mu.sum(dim=0)
        return aggregate_mu,aggregate_var
    
    def forward(self, x_list, mask=None):
        # 先将多视图数据x_list 输入inference模块，计算出每一个view的特征的潜在均值和方差
        uniview_mu_list, uniview_sca_list, mu_s_list, sca_s_list, mu_p_list, sca_p_list = self.inference_z(x_list)
        z_mu = torch.stack(uniview_mu_list,dim=0) # [v n d]
        z_sca = torch.stack(uniview_sca_list,dim=0) # [v n d]
        if torch.sum(torch.isnan(z_mu)).item() > 0:
            print("z:nan")
            pass
        # if self.training:
        #     z_mu = fill_with_label(label_embedding_mu,label,z_mu,mask)
        #     z_sca = fill_with_label(label_embedding_var,label,z_sca,mask)
        # 通过poe融合技术将先前计算出来的每一个view的均值和方差进行融合
        # fusion_mu, fusion_sca = self.poe_aggregate(z_mu, z_sca, mask)
        # # 进行重参数化，从融合分布采样潜在变量z_sample
        # z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca,times=10)
        
        # 采样各视图的共享和私有表示
        z_sample_list_s = []
        for i in range(len(uniview_mu_list)):
            z_sample_view_s = gaussian_reparameterization_var(mu_s_list[i], sca_s_list[i], times=5)
            z_sample_list_s.append(z_sample_view_s)
        
        z_sample_list_p = []
        for i in range(len(uniview_mu_list)):
            z_sample_view_p = gaussian_reparameterization_var(mu_p_list[i], sca_p_list[i], times=5)
            z_sample_list_p.append(z_sample_view_p)
        
        # 使用generation从采样的z中重构视图，结果是一个list，是每个视图的表示

        xr_s_list = []
        xr_p_list = []
        for v in range(self.num_views):
            reconstruct_x_s = self.px_generation_s[v](z_sample_list_s[v])
            xr_s_list.append(reconstruct_x_s)
        for v in range(self.num_views):
            reconstruct_x_p = self.px_generation_p[v](z_sample_list_p[v])
            xr_p_list.append(reconstruct_x_p)

        # 计算互信息损失 - 使用Independent分布来处理多维情况  
        # 这一项是对应解耦文章的最后优化问题的I(x_s,y_s) 
        I_mutual_s = 0
        # 遍历所有可能的视图对(i,j)，其中i≠j
        # for i in range(self.num_views):
        #     for j in range(self.num_views):
        #         if i != j:  # 不计算自身与自身的互信息
        #             # 使用第i个视图的共享表示和第j个视图的私有表示计算互信息
        #             mi_value1 = self.mi_estimator[i](z_sample_list_s[i], z_sample_list_s[j])
        #             mi_value2 = self.mi_estimator[j](z_sample_list_s[j], z_sample_list_s[i])
        #             I_mutual_s += mi_value1.mean() + mi_value2.mean()

        pos_I_y_zxp_mean = 0
        # 这一项是对应解耦文章的最后优化问题的I(x_p,y_p) 
        # 遍历所有可能的视图对(i,j)，其中i≠j
        # for i in range(self.num_views):
        #     # 为视图i创建私有分布
        #     sca_p_positive_i = F.softplus(sca_p_list[i]) + 1e-6
        #     p_z_yp_given_y_i = torch.distributions.Normal(loc=mu_p_list[i], scale=torch.sqrt(sca_p_positive_i))
        #     p_z_yp_given_y_i = torch.distributions.Independent(p_z_yp_given_y_i, 1)
            
        #     # 视图i的私有表示的对数概率
        #     i_log_prob = p_z_yp_given_y_i.log_prob(z_sample_list_p[i]).mean()
            
        #     for j in range(self.num_views):
        #         if i != j:  # 不计算自身与自身的互信息
        #             # 计算视图j的先验对数概率
        #             j_prior_log_prob = self.prior2.log_prob(z_sample_list_p[j]).mean()
                    
        #             # 计算视图i的私有表示与视图j的先验之间的互信息
        #             pos_I_y_zxp = i_log_prob - j_prior_log_prob
        #             pos_I_y_zxp_mean += pos_I_y_zxp
        # 遍历所有可能的视图对(i,j)，其中i≠j
        # for i in range(self.num_views):
        #     # 为视图i创建私有表示的对数概率
        #     sca_p_positive_i = F.softplus(sca_p_list[i]) + 1e-6
            
        #     # 手动计算视图i的私有表示的对数概率
        #     i_log_prob = manual_gaussian_log_prob_stable(
        #         z_sample_list_p[i], 
        #         mu_p_list[i], 
        #         sca_p_positive_i
        #     ).mean()
            
        #     for j in range(self.num_views):
        #         if i != j:  # 不计算自身与自身的互信息
        #             # 计算视图j的先验对数概率（使用标准正态分布）
        #             j_prior_log_prob = manual_gaussian_log_prob_stable(
        #                 z_sample_list_p[j], 
        #                 self.mu2.expand_as(z_sample_list_p[j]), 
        #                 self.sigma.expand_as(z_sample_list_p[j]) ** 2
        #             ).mean()
                    
        #             # 计算视图i的私有表示与视图j的先验之间的互信息
        #             pos_I_y_zxp = i_log_prob - j_prior_log_prob
        #             pos_I_y_zxp_mean += pos_I_y_zxp
        
        # 总互信息损失
        pos_beta_I = pos_I_y_zxp_mean 
        
        return uniview_mu_list, uniview_sca_list,  xr_s_list, xr_p_list, pos_beta_I, I_mutual_s, z_sample_list_s, z_sample_list_p, mu_s_list, mu_p_list, sca_s_list, sca_p_list
