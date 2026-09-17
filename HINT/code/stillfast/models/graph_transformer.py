import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

import math
from torch.autograd import Variable


import sys
sys.path.append("/data/zst/ganov2/stillfast/models")
from GAT import GAT



class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe = pe.to("cuda")
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + Variable(self.pe[:, :x.size(1)],requires_grad = False)
        b,k,d = x.shape
        t = k-2
        x[:,2:,:] = x[:,2:,:] + self.pe[:,:t,:]
        x[:,:2,:] = x[:,:2,:] + self.pe[:,t:t+2,:]
        return x



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



def get_edge(n):
    assert n%2 ==0
    n = int(n)
    src = []
    dst = []
    for i in range(2,n,2):
        src.append(i)
        dst.append(i+1)
        src.append(i+1)
        dst.append(i)
        if i+2 in range(2,n,2):
            src.append(i)
            dst.append(i+2)
    src.append(n-2)
    dst.append(0)
    src.append(0)
    dst.append(1)
    src.append(1)
    dst.append(0)
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    res = torch.stack([src,dst],dim=0)
    return res

class TimeBasedRoPE(nn.Module):
    def __init__(self, feature_dim, max_time=20.0, time_interval=0.5, base=10000):
        """
        Args:
            feature_dim: 输入特征维度 (对应tensor的最后一维)
            max_time: 最大时间限制（秒）
            time_interval: 时间间隔（秒）
            base: 频率基数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.base = base
        self.time_interval = time_interval
        
        # 计算最大时间步数（20/0.5=40个间隔）
        self.max_time_steps = int(max_time / time_interval) + 1
        
        # 预计算频率分量（仅计算feature_dim//2个频率）
        theta = 1.0 / (base ** (torch.arange(0, feature_dim, 2).float() / feature_dim))
        self.register_buffer('theta', theta, persistent=False)
        
        # 预计算所有可能时间点的cos/sin缓存
        time_points = torch.arange(0, max_time + time_interval, time_interval)
        freqs = torch.einsum('i,j->ij', time_points, theta)
        emb = torch.cat([freqs, freqs], dim=-1)  # [time_steps, feature_dim]
        
        self.register_buffer('cos_cache', emb.cos()[None, None, :, :], persistent=False)  # [1,1,time_steps,dim]
        self.register_buffer('sin_cache', emb.sin()[None, None, :, :], persistent=False)  # [1,1,time_steps,dim]
    
    def _rotate_half(self, x):
        """将后半个特征维度旋转"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x, absolute_times):
        """
        Args:
            x: 输入tensor [batch, seq_len, feature_dim]
            absolute_times: 绝对时间矩阵 [batch, seq_len]（单位：秒）
        Returns:
            编码后的tensor [batch, seq_len, feature_dim]
        """
        batch_size, seq_len = x.shape[:2]

        # 由于得预测cls token在最前main并且绝对时间不确定，因此暂时直接使用max_time
        absolute_times = [[max_time, max_time] + x[:-2] for x in absolute_times]
        
        # 将时间转换为时间步索引（0.5秒为间隔）
        time_indices = (absolute_times / self.time_interval).round().long()  # [batch, seq_len]
        time_indices = time_indices.clamp(max=self.max_time_steps - 1)  # 限制不超过最大值
        
        # 获取对应的cos/sin值
        cos_values = self.cos_cache[:, :, time_indices]  # [batch, seq_len, 1, feature_dim]
        sin_values = self.sin_cache[:, :, time_indices]  # [batch, seq_len, 1, feature_dim]
        
        # 应用旋转公式
        rotated = x * cos_values.squeeze(-2) + self._rotate_half(x) * sin_values.squeeze(-2)
        return rotated

class GATAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 128, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.time_embedding = TimeBasedRoPE(dim)
        self.gat = GAT(num_of_layers=1,
                        num_heads_per_layer=[4],
                        num_features_per_layer=[512,512],
                        add_skip_connection=True,
                        bias=False,
                        dropout=0.3,
                        log_attention_weights=False)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.add_weight = nn.Parameter(torch.tensor(0.5))
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask, timestamps=None):
        x = self.norm(x)
        mask = mask.to(x.device)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # TODO：add Rope
        if timestamps is not None:
            q = self.time_embedding(q, timestamps)
            k = self.time_embedding(k, timestamps)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        ## add gat to att
        b,n,d = x.shape
        ## todo
        gat_attention = torch.zeros(b, self.heads, n, n, 
                          device=x.device, 
                          dtype=x.dtype) 
        for i in range(b):
            node_features = x[i]
            len_nodes = int(mask[i].sum(dim=-1).item())
            edges = get_edge(len_nodes).to(x.device)
            gat_atts = self.gat((node_features[:len_nodes],edges))
            gat_attention[i, :, :len_nodes, :len_nodes] = gat_atts
        assert gat_attention.shape == dots.shape
        gat_attention = F.layer_norm(gat_attention, [gat_attention.shape[-1]])
        # b h n n


        # # keshihua
        # if dots.shape[-1]>=30 and dots.shape[-1]<=40:
        #     aaaa = 1
        #     dots_avg = dots[:,0,:,:]
        #     gat_avg = gat_attention[:,0,:,:]
        #     plot_attention_heatmaps(dots_avg,gat_avg)


        combined_attention = (1-self.add_weight)* dots + self.add_weight * gat_attention

        if mask is not None:
            b, n = mask.shape
            mask = mask.bool()
            mask = mask.unsqueeze(1).unsqueeze(2).expand(-1,self.heads,n,-1)
            combined_attention = combined_attention.masked_fill(mask == 0, float('-inf'))
        attn = self.attend(combined_attention)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 128, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.gat = GAT(num_of_layers=3,
                        num_heads_per_layer=[4,4,1],
                        num_features_per_layer=[512,256,128,64],
                        add_skip_connection=True,
                        bias=True,
                        dropout=0.4,
                        log_attention_weights=False)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.add_weight = nn.Parameter(torch.tensor(0.1))
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask, timestamps):
        x = self.norm(x)
        mask = mask.to(x.device)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # TODO：add Rope
        if timestamps is not None:
            q = self.time_embedding(q, timestamps)
            k = self.time_embedding(k, timestamps)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        ## add gat to att

        if mask is not None:
            b, n = mask.shape
            mask = mask.bool()
            mask = mask.unsqueeze(1).unsqueeze(2).expand(-1,self.heads,n,-1)
            dots = dots.masked_fill(mask == 0, float('-inf'))
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i==0:
                self.layers.append(nn.ModuleList([
                GATAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
            else:
                self.layers.append(nn.ModuleList([
                    GATAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(dim, mlp_dim, dropout = dropout)
                ]))

    def forward(self, x, mask, timestamps):
        x = self.norm(x)
        for attn, ff in self.layers:
            res = x
            x = attn(x, mask, timestamps) + res
            res = x
            x = ff(x) + res

        return self.norm(x)
    
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from GAT import GAT
class Temporal_encoder(nn.Module):
    def __init__(self,cfg,max_len=1500):
        super().__init__()

        self.dim = cfg['model']['dim']
        self.depth = cfg['model']['depth']
        self.heads = cfg['model']['heads']
        self.dim_head = cfg['model']['dim_head']
        self.mlp_dim = cfg['model']['mlp_dim']
        self.dropout = cfg['model']['dropout']
        self.max_len = max_len



        self.verb_cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.noun_cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.transformer = Transformer(self.dim,self.depth,self.heads,self.dim_head,self.mlp_dim,self.dropout)
        self.position_embed = PositionalEncoding(self.dim,self.max_len)
        self.type_embedding = nn.Embedding(2, self.dim)
        #self.GNN = GAT(cfg)
        self.max_len = max_len

    def forward(self,x,batch_mask, timestamps=None):
        '''
        x:batch,len,dim
        mask:batch,len
        '''
        input = []
        masks = []
        for batch_item in x:
            input.append(batch_item)
        padded_input = pad_sequence(input, batch_first=True, padding_value=0)

        for mask_item in batch_mask:
            masks.append(mask_item)
    
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

        b, t, d = padded_input.shape

        verb_cls = self.verb_cls_token.expand(b, 1, -1)  # (B, 1, D)
        noun_cls = self.noun_cls_token.expand(b, 1, -1)  # (B, 1, D)


        input = torch.cat([verb_cls, noun_cls, padded_input], dim=1)
        assert input.shape == (b,t+2,d)

        padding_mask = torch.ones((b, 2), device=padded_masks.device)
        masks = torch.cat([padding_mask, padded_masks], dim=1)

        assert input.shape[0]==masks.shape[0]
        assert masks.shape == (b,t+2)


        input = self.position_embed(input)
        type_0 = self.type_embedding(torch.tensor([0], dtype=torch.long,device=input.device)).unsqueeze(0)
        type_1 = self.type_embedding(torch.tensor([1], dtype=torch.long,device=input.device)).unsqueeze(0) 
        input[:,0::2,:] = input[:,0::2,:] + type_0
        input[:,1::2,:] = input[:,1::2,:] + type_1



        output = self.transformer(input, masks, timestamps)
        verb_cls_feature = output[:, 0]   # (B, D)
        noun_cls_feature = output[:, 1]   # (B, D)
        return verb_cls_feature, noun_cls_feature
    

class AnticipationModel(nn.Module):
    def __init__(self, base_encoder, dim, num_verb_classes, num_noun_classes):
        super().__init__()
        self.encoder = base_encoder
        self.verb_classifier = nn.Linear(dim, num_verb_classes)
        self.noun_classifier = nn.Linear(dim, num_noun_classes)

    def forward(self, x, mask, graph_mask=None, timestamps=None):
        verb_cls_feature, noun_cls_feature = self.encoder(x, mask, timestamps)
        verb_logits = self.verb_classifier(verb_cls_feature)
        noun_logits = self.noun_classifier(noun_cls_feature)
        return verb_logits, noun_logits,verb_cls_feature, noun_cls_feature




# from GAT import *
class Temporal_GAT_encoder(nn.Module):
    def __init__(self,cfg,max_len=1500):
        super().__init__()

        self.dim = cfg['model']['dim']
        self.depth = cfg['model']['depth']
        self.heads = cfg['model']['heads']
        self.dim_head = cfg['model']['dim_head']
        self.mlp_dim = cfg['model']['mlp_dim']
        self.dropout = cfg['model']['dropout']
        self.max_len = max_len
        self.gnn = GAT(num_of_layers=cfg['gat']['num_of_layers'],
                       num_heads_per_layer=cfg['gat']['num_heads_per_layer'],
                       num_features_per_layer=cfg['gat']['num_features_per_layer'],
                       add_skip_connection=cfg['gat']['add_skip_connection'],
                       bias=cfg['gat']['bias'],
                       dropout=cfg['gat']['dropout'],
                       layer_type=cfg['gat']['layer_type'],
                       log_attention_weights=cfg['gat']['log_attention_weights'])


        self.verb_cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.noun_cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.transformer = Transformer(self.dim,self.depth,self.heads,self.dim_head,self.mlp_dim,self.dropout)
        self.position_embed = PositionalEncoding(self.dim,self.max_len)
        self.type_embedding = nn.Embedding(2, self.dim)
        #self.GNN = GAT(cfg)
        self.max_len = max_len

    def forward(self,x,batch_mask):
        '''
        x:batch,len,dim
        mask:batch,len
        '''
        input = []
        masks = []
        for batch_item in x:
            input.append(batch_item)
        padded_input = pad_sequence(input, batch_first=True, padding_value=0)

        for mask_item in batch_mask:
            masks.append(mask_item)
    
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

        b, t, d = padded_input.shape

        verb_cls = self.verb_cls_token.expand(b, 1, -1)  # (B, 1, D)
        noun_cls = self.noun_cls_token.expand(b, 1, -1)  # (B, 1, D)

        input = torch.cat([verb_cls, noun_cls, padded_input], dim=1)
        assert input.shape == (b,t+2,d)

        padding_mask = torch.ones((b, 2), device=padded_masks.device)
        masks = torch.cat([padding_mask, padded_masks], dim=1)

        assert input.shape[0]==masks.shape[0]
        assert masks.shape == (b,t+2)


        input = self.position_embed(input)
        type_0 = self.type_embedding(torch.tensor([0], dtype=torch.long,device=input.device)).unsqueeze(0)
        type_1 = self.type_embedding(torch.tensor([1], dtype=torch.long,device=input.device)).unsqueeze(0) 
        input[:,0::2,:] = input[:,0::2,:] + type_0
        input[:,1::2,:] = input[:,1::2,:] + type_1


        output = self.transformer(input,masks)
        verb_cls_feature = output[:, 0]   # (B, D)
        noun_cls_feature = output[:, 1]   # (B, D)
        return verb_cls_feature, noun_cls_feature










    
# if __name__ == "__main__":
#     x = []
#     masks = []
#     mask1 = torch.zeros(500)
#     mask2 = torch.zeros(500)
#     mask1[:2] = 1
#     mask2[:4] = 1
#     masks.append(mask1)
#     masks.append(mask2)
#     x.append([torch.randn(12288) for _ in range(2)])
#     x.append([torch.randn(12288) for _ in range(4)])
#     model = Temporal_encoder(depth=1,dim=12288,dim_head=12288,heads=1,mlp_dim=2048,dropout=0.1)
#     print(model(x,masks).shape)




# class GATAttention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 128, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
#         self.gat = GAT(num_of_layers=2,
#                         num_heads_per_layer=[1,1],
#                         num_features_per_layer=[512,512,512],
#                         add_skip_connection=True,
#                         bias=True,
#                         dropout=0.1,
#                         log_attention_weights=False)
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.norm = nn.LayerNorm(dim)
#         self.add_weight = nn.Parameter(torch.tensor(0.5))
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)
        

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x,mask):
#         x = self.norm(x)
#         mask = mask.to(x.device)
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         ## add gat to att
#         b,n,d = x.shape
#         ## todo
#         gat_attention = torch.zeros(b, self.heads, n, n, 
#                           device=x.device, 
#                           dtype=x.dtype) 
#         for i in range(b):
#             node_features = x[i]
#             len_nodes = int(mask[i].sum(dim=-1).item())
#             edges = get_edge(len_nodes).to(x.device)
#             gat_atts = self.gat((node_features[:len_nodes],edges))
#             gat_attention[i, :, :len_nodes, :len_nodes] = gat_atts
#         assert gat_attention.shape == dots.shape
#         combined_attention = self.add_weight * dots + (1 - self.add_weight) * gat_attention

#         if mask is not None:
#             b, n = mask.shape
#             mask = mask.bool()
#             mask = mask.unsqueeze(1).unsqueeze(2).expand(-1,self.heads,n,-1)
#             combined_attention = combined_attention.masked_fill(mask == 0, float('-inf'))
#         attn = self.attend(combined_attention)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


        

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)




if __name__ == "__main__":
    print(get_edge(6))