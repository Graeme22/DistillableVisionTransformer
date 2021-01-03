# PyTorch Vision Transformers with Distillation
Based on the paper "[Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)".

This repository will allow you to use distillation techniques with vision transformers in PyTorch. Most importantly, you can use pretrained models for the teacher, the student, or even both! My motivation was to use transfer learning to decrease the amount of resources it takes to train a vision transformer.

### Quickstart

Install with `pip install distillable_vision_transformer` and load a pretrained transformer with:

```
from distillable_vision_transformer import DistillableVisionTransformer
model = DistillableVisionTransformer.from_pretrained('ViT-B_16')
```

### Installation

Install via pip:

```
pip install distillable_vision_transformer
```

Or install from source:

```
git clone https://github.com/Graeme22/DistillableVisionTransformer.git
cd DistillableVisionTransformer
pip install -e .
```

### Usage

Load a model architecture:

```
from distillable_vision_transformer import DistillableVisionTransformer
model = DistillableVisionTransformer.from_name('ViT-B_16')
```

Load a pretrained model:

```
from distillable_vision_transformer import DistillableVisionTransformer
model = DistillableVisionTransformer.from_pretrained('ViT-B_16')
```

Default hyper parameters:

| Param\Model       | ViT-B_16 | ViT-B_32 | ViT-L_16 | ViT-L_32 |
| ----------------- | -------- | -------- | -------- | -------- |
| image_size        | 384, 384 | 384, 384 | 384, 384 | 384, 384 |
| patch_size        | 16       | 32       | 16       | 32       |
| emb_dim           | 768      | 768      | 1024     | 1024     |
| mlp_dim           | 3072     | 3072     | 4096     | 4096     |
| num_heads         | 12       | 12       | 16       | 16       |
| num_layers        | 12       | 12       | 24       | 24       |
| num_classes       | 1000     | 1000     | 1000     | 1000     |
| attn_dropout_rate | 0.0      | 0.0      | 0.0      | 0.0      |
| dropout_rate      | 0.1      | 0.1      | 0.1      | 0.1      |

If you need to modify these hyperparameters, just overwrite them:

```
model = DistillableVisionTransformer.from_name('ViT-B_16', patch_size=64, emb_dim=2048, ...)
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.
