# MODIFIED FROM
# https://github.com/tczhangzhi/VisionTransformer-Pytorch/blob/main/vision_transformer_pytorch/model.py
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/distill.py

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import load_pretrained_weights, get_model_params

VALID_MODELS = ('ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32')

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, emb_dim))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out

class DistillationEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim):
        super(DistillationEmbs, self).__init__()
        self.dis_embedding = nn.Parameter(torch.randn(1, num_patches + 2, emb_dim))

    def forward(self, x):
        out = x + self.dis_embedding
        return out

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out

class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = self.out(out, dims=([2, 3], [0, 1]))
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out

class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        # distillation token
        self.dis_embedding = DistillationEmbs(num_patches, emb_dim)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        out = self.pos_embedding(x)
        out = self.dis_embedding(out)

        for layer in self.encoder_layers:
          out = layer(out)

        out = self.norm(out)
        return out

class DistillableVisionTransformer(nn.Module):
    '''
    A vision transformer with an added distillation token and a modified forward pass.
    To train it, use a DistillationTrainer. Pretrained models are available.
    Most easily loaded with the from_name() or from_pretrained() methods.
    References:
        [1] https://arxiv.org/abs/2010.11929 (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)
        [2] https://arxiv.org/abs/2012.12877 (Training data-efficient image transformers & distillation through attention)
    Example:
        >>> import torch
        >>> from distillable_vision_transformer import DistillableVisionTransformer
        >>> inputs = torch.rand(1, 3, 384, 384)
        >>> model = DistillableVisionTransformer.from_pretrained('ViT-B_16')
        >>> model.eval()
        >>> outputs, _ = model(inputs)
    '''
    def __init__(self, in_channels=3, image_size=(384, 384), patch_size=16, emb_dim=768, mlp_dim=3072, num_heads=12, num_layers=12, num_classes=1000, attn_dropout_rate=0.0, dropout_rate=0.1):
        super(DistillableVisionTransformer, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.attn_dropout_rate = attn_dropout_rate
        self.dropout_rate = dropout_rate

        self.embedding = nn.Conv2d(self.in_channels, self.emb_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self.dis_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        self.transformer = Encoder(num_patches=self.num_patches, emb_dim=self.emb_dim, mlp_dim=self.mlp_dim, num_layers=self.num_layers, num_heads=self.num_heads, dropout_rate=self.dropout_rate, attn_dropout_rate=self.attn_dropout_rate)
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    @property
    def num_patches(self):
        h, w = self.image_size
        gh, gw = h // self.patch_size, w // self.patch_size
        return gh * gw

    def forward(self, img):
        emb = self.embedding(img) # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1) # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)
        # append distillation token
        dis_token = self.dis_token.repeat(b, 1, 1)
        emb = torch.cat([emb, dis_token], dim=1)

        x = self.transformer(emb)
        feat, distill_tokens = x[:, :-1], x[:, -1]
        # run through classifier to obtain logits
        logits = self.classifier(feat[:, 0])
        # when performing inference, the tokens can be discarded
        return logits, distill_tokens

    @classmethod
    def from_name(cls, model_name, **override_params):
        '''
        Create an vision transformer model with the given architecture.
        Args:
            model_name (str): Name for vision transformer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'image_size', 'patch_size',
                    'emb_dim', 'mlp_dim',
                    'num_heads', 'num_layers',
                    'num_classes', 'attn_dropout_rate',
                    'dropout_rate', 'in_channels'
        Returns:
            A distillable vision transformer model.
        '''
        cls._check_model_name_is_valid(model_name)
        params = get_model_params(model_name)
        params.update(override_params)
        model = cls(**params)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, **override_params):
        '''
        Create an vision transformer model with the given architecture.
        Args:
            model_name (str): Name for vision transformer.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the internet.
            override_params (other keyword params):
                Params to override model's global_params.
                Optional key:
                    'image_size', 'patch_size',
                    'emb_dim', 'mlp_dim',
                    'num_heads', 'num_layers',
                    'num_classes', 'attn_dropout_rate',
                    'dropout_rate', 'in_channels'
        Returns:
            A pretrained vision transformer model.
        '''
        cls._check_model_name_is_valid(model_name)
        model = cls.from_name(model_name, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(model.num_classes == 1000))
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        '''
        Validates model name.
        Args:
            model_name (str): Name for vision transformer.
        Returns:
            bool: Is a valid name or not.
        '''
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

class DistillationTrainer(nn.Module):
    '''
    Wrapper to train a DistillableVisionTransformer object.
    Given a pretrained image classifier (the "teacher"), it trains its own MLP to output distilled logits.
    These distilled logits can then be used to calculate a DistilledLoss, which optimizes the student by
    minimizing the KL divergence between its distilled logits and the teacher's own logits.
    Args:
        teacher (torch.nn.Module): Pretrained classifier that provides a baseline for the student.
        student (DistillableVisionTransformer): Transformer to be trained, using the teacher as a baseline.
    '''
    def __init__(self, teacher, student):
        super().__init__()
        assert isinstance(student, DistillableVisionTransformer), 'student must be a DistillableVisionTransformer'

        self.teacher = teacher
        self.student = student
        self.dim = student.emb_dim
        self.num_classes = student.num_classes
        
        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

    def forward(self, img):
        with torch.no_grad():
            teacher_logits = self.teacher(img)
        student_logits, distill_tokens = self.student(img)
        distill_logits = self.distill_mlp(distill_tokens)

        return teacher_logits, student_logits, distill_logits

class DistilledLoss(nn.Module):
    '''
    Intended for use with a DistillationTrainer.
    Combines vanilla cross entropy loss with a modified form of KL divergence loss.
    Attempts to minimize the KL divergence between the student and distilled logits
    while maintaining an emphasis on predicting the true labels with cross entropy.
    '''
    def __init__(self, alpha=0.5, temperature=1.0):
        super(DistilledLoss, self).__init__()
        # controls the weight of student logits. at alpha = 1.0, reduces to vanilla cross entropy.
        self.alpha = alpha
        # scaling factor of distilled loss.
        self.temperature = temperature

    def forward(self, teacher_logits, student_logits, distill_logits, labels):
        loss = F.cross_entropy(student_logits, labels)
        # our goal is to minimize the KL divergence between the distilled logits and the teacher logits
        distill_loss = F.kl_div(F.log_softmax(distill_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1).detach(), reduction='batchmean')
        distill_loss *= self.temperature ** 2

        # final loss is a weighted combination of vanilla cross entropy on the student logits
        # and the KL divergence between the distilled and teacher logits.
        return loss * self.alpha + distill_loss * (1 - self.alpha)
