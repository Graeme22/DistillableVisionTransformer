# MODIFIED FROM
# https://github.com/tczhangzhi/VisionTransformer-Pytorch/blob/main/vision_transformer_pytorch/utils.py

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

# pretrained weights from v1.0.1
url_map = {
    'ViT-B_16': 'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-B_16_imagenet21k_imagenet2012.pth',
    'ViT-B_32': 'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-B_32_imagenet21k_imagenet2012.pth',
    'ViT-L_16': 'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-L_16_imagenet21k_imagenet2012.pth',
    'ViT-L_32': 'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-L_32_imagenet21k_imagenet2012.pth'
}

def get_model_params(model_name):
    '''
    Gets parameters for given vision transformer model.
    Args:
        model_name (str): Model name to be queried.
    Returns:
        Dictionary containing parameters
    '''
    params_dict = {
        'ViT-B_16': ((384, 384), 16, 768, 3072, 12, 12, 1000, 0.0, 0.1),
        'ViT-B_32': ((384, 384), 32, 768, 3072, 12, 12, 1000, 0.0, 0.1),
        'ViT-L_16': ((384, 384), 16, 1024, 4096, 16, 24, 1000, 0.0, 0.1),
        'ViT-L_32': ((384, 384), 32, 1024, 4096, 16, 24, 1000, 0.0, 0.1)
    }
    image_size, patch_size, emb_dim, mlp_dim, num_heads, num_layers, num_classes, attn_dropout_rate, dropout_rate = params_dict[model_name]

    return dict(
        image_size=image_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        attn_dropout_rate=attn_dropout_rate,
        dropout_rate=dropout_rate
    )

def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True):
    '''
    Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): The whole model of vision transformer.
        model_name (str): Model name of vision transformer.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
    '''
    if weights_path != None:
        state_dict = torch.load(weights_path)
    else:
        state_dict = model_zoo.load_url(url_map[model_name])
    if not load_fc:
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')

    # concatenate empty space for distillation token
    state_dict['transformer.pos_embedding.pos_embedding'] = torch.cat([state_dict['transformer.pos_embedding.pos_embedding'], torch.zeros(1, 1, model.emb_dim)], dim=1)
    model.load_state_dict(state_dict, strict=False)

    print('Loaded pretrained weights for {}'.format(model_name))
