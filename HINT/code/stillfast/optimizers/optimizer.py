import torch

from . import lr_policy as lr_policy

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ('backbone.fast_backbone.cls_token', 'backbone.fast_backbone.mask_token',
                    'backbone.fast_backbone.pos_embed', 'backbone.fast_backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.fast_backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.fast_backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.fast_backbone.blocks') or var_name.startswith(
            'backbone.fast_backbone.layers'):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1
    else:
        return num_max_layer - 1


def construct_optimizer(model, cfg):
    parameter_groups = {}
    # num_layers = cfg.get('num_layers') + 2
    num_layers = cfg.SOLVER.NUM_LAYERS + 2
    layer_decay_rate = cfg.SOLVER.LAYER_DECAY_RATE
    print('Build LayerDecayOptimizerConstructor %f - %d' %
          (layer_decay_rate, num_layers))
    base_lr = cfg.SOLVER.BASE_LR
    base_wd = cfg.SOLVER.WEIGHT_DECAY

    bn_params = []
    fast_params = []
    still_params = []
    non_bn_parameters = []

    for name, param in model.named_parameters():
        if "bn" in name and "fast_backbone" not in name:
            bn_params.append(param)
        elif "fast_backbone" in name:
            fast_params.append(param)
        elif "still_backbone" in name:
            still_params.append(param)
        else:
            non_bn_parameters.append(param)
    fast_param_ids = {id(p) for p in fast_params}
    # Apply layer-wise decay for fast_params
    for name, param in model.named_parameters():
        if id(param) not in fast_param_ids:
            continue

        if len(param.shape) == 1 or name.endswith('.bias') or name in (
                'pos_embed', 'cls_token', 'visual_embed'):
            group_name = 'no_decay'
            this_weight_decay = 0.
        else:
            group_name = 'decay'
            this_weight_decay = base_wd

        layer_id = get_num_layer_for_vit(name, num_layers)
        group_name = 'layer_%d_%s' % (layer_id, group_name)

        if group_name not in parameter_groups:
            scale = layer_decay_rate ** (num_layers - layer_id - 1)

            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * base_lr,
            }

        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)
    # Combine with other parameter groups
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": still_params, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, "lr": cfg.SOLVER.BASE_LR*0.1},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    ] + list(parameter_groups.values())
    # Check all parameters will be passed into optimizer.
    all_params = set(p for group in optim_params for p in group['params'])
    assert len(list(model.parameters())) == len(all_params), "Mismatch in parameter count."

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'adamw':
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        # pass
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr