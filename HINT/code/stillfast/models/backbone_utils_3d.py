from typing import Optional, List, Dict
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from stillfast.models.feature_pyramid_network_3d import FeaturePyramidNetwork3D, LastLevelMaxPool3D
from torch import nn
from torch.nn import functional as F
import torch
from torch import Tensor
from stillfast.ops.misc import TemporalCausalConv3D
from stillfast.models.vision_transformer import vit_base_patch16_224_umt_k400, vit_large_patch16_224
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
import clip
# import vit-1b from avion

import sys

sys.path.append("/data/EgoHOD/evaluation")
from bulid_hod_backbone import ego_backbone
# from avion.models.internvideo2.internvideo2_clip_vision import InternVideo2
# from avion.models.model_clip import CLIP
# from avion.models.transformer import TextTransformer
# print("import VisionTransformer from avion")





def InternVideoCLIP_VIT_g_14_ego4dv1_howtoego_pretrained_f8():
    model = ego_backbone()
    print("loading model ==>")
    data = torch.load("/data/egohod_ckpts/large_best.pt", map_location='cpu')
    data_dict = {}
    model_dict = model.state_dict()  

    for n, v in data["state_dict"].items():
        new_name = n[7:]  
        if new_name == "visual.temporal_embedding":
            print(f"Interpolating temporal_embedding from {v.shape} to {model_dict[new_name].shape}")
            # 插值: [4, 1024] -> [16, 1024]
            # 加上 batch 和 channel 维度: [1, 1024, 4] -> 插值 -> [1, 1024, 16]
            v_interp = F.interpolate(v.unsqueeze(0).permute(0, 2, 1), size=model_dict[new_name].shape[0], mode='linear', align_corners=True)
            v = v_interp.permute(0, 2, 1).squeeze(0)  # 变成 [16, 1024]
    
        if new_name in model_dict and v.shape == model_dict[new_name].shape:
            data_dict[new_name] = v
        else:
            print(v.shape)
            print(model_dict[new_name].shape)

    missing_keys, unexpected_keys = model.load_state_dict(data_dict, strict=False)
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)

    return model
    


def replace_module(m, replace_type, replace_func):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == replace_type:
            setattr(m, attr_str, TemporalCausalConv3D.from_conv3d(target_attr))
    for n, ch in m.named_children():
        replace_module(ch, replace_type, replace_func)

def build_clean_3d_backbone(
    backbone_name: str, 
    pretrained: bool,
    temporal_causal_conv3d: bool = False,
):
    if backbone_name not in ['slow_r50', 'x3d_l', 'x3d_m', 'r2plus1d_r50', 'vit', 'vit-1b']:
        raise ValueError(f"Backbone {backbone_name} is not supported with 3D models")
    if backbone_name == "vit":
        backbone = torch.hub.load("facebookresearch/pytorchvideo", model=backbone_name, pretrained=pretrained)
        del backbone.blocks[5]
    elif backbone_name == "vit-1b":
        backbone = InternVideoCLIP_VIT_g_14_ego4dv1_howtoego_pretrained_f8()
    else:
        backbone = vit_base_patch16_224_umt_k400(ckpt_path="/mnt/petrelfs/chenguo/workspace/w/workspace/EgoAnticipator/work_dirs/anti_pretrain_ego4d_v0_vitb_umt_k400/iter_51000.pth")
    if backbone_name in ['slow_r50', 'r2plus1d_r50']:
        channels_list = [256, 512, 1024, 2048]
    elif backbone_name in ['x3d_l', 'x3d_m']:
        channels_list = [24, 48, 96, 192]
    elif backbone_name in ['vit-1b']:
        channels_list = [512, 512, 512, 512]
    backbone.channels = channels_list

    if temporal_causal_conv3d:
        replace_module(backbone, nn.Conv3d, TemporalCausalConv3D.from_conv3d)
        print('Replaced all Conv3d in the fast backbone with TemporalCausalConv3D keeping the same weights')

    return backbone



# from typing import Optional, List, Dict
# from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
# from torchvision.models.detection.backbone_utils import BackboneWithFPN
# from stillfast.models.feature_pyramid_network_3d import FeaturePyramidNetwork3D, LastLevelMaxPool3D
# from torch import nn
# from torch.nn import functional as F
# import torch
# from torch import Tensor
# from stillfast.ops.misc import TemporalCausalConv3D
# from stillfast.models.vision_transformer import vit_base_patch16_224_umt_k400, vit_large_patch16_224

# # import vit-1b from avion

# import sys

# sys.path.append("/data/EgoVideo/cvpr-2024/AVION")
# from avion.models.internvideo2.internvideo2_clip_vision import InternVideo2
# from avion.models.model_clip import CLIP
# from avion.models.transformer import TextTransformer
# print("import VisionTransformer from avion")

# def new_forward(self, x,return_f3d=True):
#     x = self.visual(x, return_f3d=True)
#     return  x

# CLIP.forward = new_forward

# def InternVideoCLIP_VIT_g_14_ego4dv1_howtoego_pretrained_f8(
#     freeze_temperature=False,
#     use_grad_checkpointing=False,
#     use_bidirectional_lm=False,
#     context_length=77,
#     vocab_size=49408,
#     patch_dropout=0.,
#     drop_path_rate=0.,
#     num_frames=8,
#     use_fast_conv1=False,
#     use_flash_attn=False,
#     project_embed_dim=512,
#     pretrain_zoo='openai',
#     **kwargs
# ):
#     # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
#     vision_model = InternVideo2(
#         in_chans=3,
#         patch_size=14,
#         img_size=224,
#         qkv_bias=False,
#         drop_path_rate=drop_path_rate,
#         head_drop_path_rate=0.,
#         embed_dim=1408,
#         num_heads=16,
#         mlp_ratio=48/11,
#         init_values=0.1,
#         qk_normalization=True,
#         depth=40,
#         use_flash_attn=False,
#         use_fused_rmsnorm=False,
#         use_fused_mlp=False,
#         fused_mlp_heuristic=1,
#         attn_pool_num_heads=16,
#         clip_embed_dim=768,
#         layerscale_no_force_fp32=True,
#         num_frames=num_frames,
#         tubelet_size=1,
#         sep_pos_embed=False,
#         use_checkpoint=True,
#         checkpoint_num=40,
#         # add_fc_norm=False,
#     )
#     text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
#     model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature, vision_width=768)

#     # ckpt = torch.load("/mnt/petrelfs/chenguo/pretrained_models/internvideo/intern_1b_8f.pt", map_location='cpu')
#     # state_dict = ckpt["state_dict"]
#     # new_state_dict = dict()
#     # for key, value in state_dict.items():
#     #     if key.startswith("module."):
#     #         new_state_dict[key[7:]] = value

#     # info =  model.load_state_dict(new_state_dict, strict=False)    
#     # print(info)    

#     return model
    


# def replace_module(m, replace_type, replace_func):
#     for attr_str in dir(m):
#         target_attr = getattr(m, attr_str)
#         if type(target_attr) == replace_type:
#             setattr(m, attr_str, TemporalCausalConv3D.from_conv3d(target_attr))
#     for n, ch in m.named_children():
#         replace_module(ch, replace_type, replace_func)

# def build_clean_3d_backbone(
#     backbone_name: str, 
#     pretrained: bool,
#     temporal_causal_conv3d: bool = False,
# ):
#     if backbone_name not in ['slow_r50', 'x3d_l', 'x3d_m', 'r2plus1d_r50', 'vit', 'vit-1b']:
#         raise ValueError(f"Backbone {backbone_name} is not supported with 3D models")
#     if backbone_name == "vit":
#         backbone = torch.hub.load("facebookresearch/pytorchvideo", model=backbone_name, pretrained=pretrained)
#         del backbone.blocks[5]
#     elif backbone_name == "vit-1b":
#         backbone = InternVideoCLIP_VIT_g_14_ego4dv1_howtoego_pretrained_f8()
#     else:
#         backbone = vit_base_patch16_224_umt_k400(ckpt_path="/mnt/petrelfs/chenguo/workspace/w/workspace/EgoAnticipator/work_dirs/anti_pretrain_ego4d_v0_vitb_umt_k400/iter_51000.pth")
#     if backbone_name in ['slow_r50', 'r2plus1d_r50']:
#         channels_list = [256, 512, 1024, 2048]
#     elif backbone_name in ['x3d_l', 'x3d_m']:
#         channels_list = [24, 48, 96, 192]
#     elif backbone_name in ['vit-1b']:
#         channels_list = [768, 768, 768, 768]
#     backbone.channels = channels_list

#     if temporal_causal_conv3d:
#         replace_module(backbone, nn.Conv3d, TemporalCausalConv3D.from_conv3d)
#         print('Replaced all Conv3d in the fast backbone with TemporalCausalConv3D keeping the same weights')

#     return backbone