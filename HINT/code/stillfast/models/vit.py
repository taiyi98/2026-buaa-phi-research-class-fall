from fvcore.common.config import CfgNode as CN
import yaml
from stillfast.models.graph_transformer import AnticipationModel, Temporal_encoder
import torch


def load_vit():

    def load_cfg(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return CN(cfg_dict)

    config = load_cfg('/data/zst/ganov2/configs/sta/encoder.yaml')

    base_encoder = Temporal_encoder(
        cfg=config,max_len=1500
        )
    model = AnticipationModel(base_encoder, dim=config['model']['dim'], num_verb_classes=config['class_config']['num_verb_classes'], num_noun_classes=config['class_config']['num_noun_classes'])
    model = model.to("cuda")
    # ckpt_path = "/data/zst/ganov2/checkpoints_text_no_dp4_att/best_acc_model.pt"
    ckpt_path = "/data/zst/ganov2/checkpoints_text_no_dp4_gnn_v2_abation_len_400/epo30-vacc0.6412279776347716-nacc0.7501846186306572.pt"
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path), strict=True)
    print("longding the vit ckpt")
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)
    return model

