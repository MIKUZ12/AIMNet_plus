import torch
# from utils.expert import weight_sum_var, ivw_aggregate_var
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
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
        z_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
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
        self.px_generation = nn.ModuleList([Px_generation_mlp(self.z_dim, self.x_dim_list[v]) for v in range(self.num_views)])
    def inference_z(self, x_list):
        uniview_mu_list = []
        uniview_sca_list = []
        uniview_mu_s_list = []
        uniview_sca_s_list = []
        uniview_mu_p_list = []
        uniview_sca_p_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            # 每一个view都通过qz_inference提取特征
            fea = self.qz_inference[v](x_list[v])
            fea_s = self.qz_inference_s[v](x_list[v])
            fea_p = self.qz_inference_p[v](x_list[v])
            if torch.sum(torch.isnan(fea)).item() > 0:
                print("zz:nan")
                pass
            # 通过共享头生成每个特征的潜在分布均值和方差
            z_mu_v, z_sca_v = self.qz_inference_header(fea)
            z_mu_v_s, z_sca_v_s = self.qz_inference_header(fea_s)
            z_mu_v_p, z_sca_v_p = self.qz_inference_header(fea_p)
            if torch.sum(torch.isnan(z_mu_v)).item() > 0:
                print("zzmu:nan")
                pass
            if torch.sum(torch.isnan(z_sca_v)).item() > 0:
                print("zzvar:nan")
                pass
            uniview_mu_list.append(z_mu_v)
            uniview_sca_list.append(z_sca_v)
            uniview_mu_s_list.append(z_mu_v_s)
            uniview_sca_s_list.append(z_sca_v_s)
            uniview_mu_p_list.append(z_mu_v_p)
            uniview_sca_p_list.append(z_sca_v_p)





        # mu = torch.stack(z_mu)
        # sca = torch.stack(z_sca)
        
        ###POE aggregation
        # fusion_mu, fusion_sca = self.poe_aggregate(mu, sca, mask)

        ###weighted fusion
        # z = []
        # for v in range(self.num_views):
        #     z.append(gaussian_reparameterization_var(uniview_mu_list[v], uniview_sca_list[v],times=1))
        # z = torch.stack(z,dim=1) #[n v d]
        # z = z.mul(mask.unsqueeze(-1)).sum(1)
        # z = z / (mask.sum(1).unsqueeze(-1)+1e-8)
        return uniview_mu_list, uniview_sca_list, uniview_mu_s_list, uniview_sca_s_list, uniview_mu_p_list, uniview_sca_p_list
    
    def generation_x(self, z):
        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist
    
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
        fusion_mu, fusion_sca = self.poe_aggregate(z_mu, z_sca, mask)
        # 进行重参数化，从融合分布采样潜在变量z_sample
        z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca,times=10)
        
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
        xr_list = self.generation_x(z_sample)
        xr_s_list = []
        xr_p_list = []
        for v in range(self.num_views):
            reconstruct_x_s = self.px_generation_s[v](z_sample_list_s[v])
            xr_s_list.append(reconstruct_x_s)
        for v in range(self.num_views):
            reconstruct_x_p = self.px_generation_p[v](z_sample_list_p[v])
            xr_p_list.append(reconstruct_x_p)

        # 计算互信息损失 - 使用Independent分布来处理多维情况
        pos_I_y_zxp_list = []

        
        for v in range(self.num_views):
            # 为每个视图创建共享和私有分布
            sca_s_positive = F.softplus(sca_s_list[v]) + 1e-6
            sca_p_positive = F.softplus(sca_p_list[v]) + 1e-6
            p_z_ys_given_y = torch.distributions.Normal(loc=mu_s_list[v], scale=torch.sqrt(sca_s_positive))
            p_z_yp_given_y = torch.distributions.Normal(loc=mu_p_list[v], scale=torch.sqrt(sca_p_positive))
            

            # 创建独立分布，设置重要的1参数，表示最后一个维度是独立的
            p_z_ys_given_y = torch.distributions.Independent(p_z_ys_given_y, 1)
            p_z_yp_given_y = torch.distributions.Independent(p_z_yp_given_y, 1)
            

            other_views_prior_sum = 0
            for other_v in range(self.num_views):
                if other_v != v:
                    other_views_prior_sum += self.prior2.log_prob(z_sample_list_p[other_v])
            
            # 如果有其他视图，计算平均值
            if self.num_views > 1:
                other_views_prior_avg = other_views_prior_sum / (self.num_views - 1)
            else:
                # 如果只有一个视图，直接使用标准先验
                other_views_prior_avg = self.prior2.log_prob(z_sample_list_p[v])
            
            # 当前视图的私有表示 vs 其他视图的先验平均
            pos_I_y_zxp = p_z_yp_given_y.log_prob(z_sample_list_p[v]).mean() - other_views_prior_avg.mean()
            
            pos_I_y_zxp_list.append(pos_I_y_zxp.mean()) 
        
        # 计算所有视图的平均互信息损失
        pos_I_y_zxp_mean = sum(pos_I_y_zxp_list) / len(pos_I_y_zxp_list)
        
        # 总互信息损失
        pos_beta_I = pos_I_y_zxp_mean
        
        return z_sample, uniview_mu_list, uniview_sca_list, fusion_mu, fusion_sca, xr_list, xr_s_list, xr_p_list, pos_beta_I
