# PyTorch Vision Transformers with Distillation
Based on the paper "[Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)".

This repository will allow you to use distillation techniques with vision transformers in PyTorch. Most importantly, you can use pretrained models for the teacher, the student, or even both! My motivation was to use transfer learning to decrease the amount of resources it takes to train a vision transformer.

### Quickstart

Install with `pip install distillable_vision_transformer` and load a pretrained transformer with:

```python
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

```python
from distillable_vision_transformer import DistillableVisionTransformer
model = DistillableVisionTransformer.from_name('ViT-B_16')
```

Load a pretrained model:

```python
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

```python
model = DistillableVisionTransformer.from_name('ViT-B_16', patch_size=64, emb_dim=2048, ...)
```

### Training

Wrap the student (instance of `DistillableVisionTransformer`) and the teacher (any network that you want to use to train the student) with a `DistillationTrainer`:

```python
from distillable_vision_transformer import DistillableVisionTransformer, DistillationTrainer

student = DistillableVisionTransformer.from_pretrained('ViT-B_16')
trainer = DistillationTrainer(teacher=teacher, student=student) # where teacher is some pretrained network, e.g. an EfficientNet
```

For the loss function, it is recommended that you use the `DistilledLoss` class, which is a kind of hybrid between cross-entropy and KL-divergence loss.
It takes as arguments `teacher_logits`, `student_logits`, and `distill_logits`, which are obtained from the forward pass on `DistillationTrainer`, as well as the true labels `labels`.

```python
from distillable_vision_transformer import DistilledLoss

loss_fn = DistilledLoss(alpha=0.5, temperature=3.0)
loss = loss_fn(teacher_logits, student_logits, distill_logits, labels)
```

### Inference

For inference, we want to use the `DistillableVisionTransformer` instance, not its `DistillationTrainer` wrapper.

```python
import torch
from distillable_vision_transformer import DistillableVisionTransformer

model = DistillableVisionTransformer.from_pretrained('ViT-B_16')
model.eval()

inputs = torch.rand(1, 3, *model.image_size)
# we can discard the distillation tokens, as they are only needed to calculate loss
outputs, _ = model(inputs)
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.
