import torch
import torch.nn.functional as F
from torch import nn
from vision_transformer_pytorch import VisionTransformer

# classes
class DistillableVisionTransformer(VisionTransformer):
    def __init__(self, params):
        super(VisionTransformer, self).__init__(params)
        self.dim = params['dim']
        self.num_classes = params['num_classes']
        
    def forward(self, img, distill_token):
        emb = self.embedding(img)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)
        
        distill_tokens = distill_token.repeat(b, 1, 1)
        emb = torch.cat([emb, distill_tokens], dim=1)

        # transformer
        feat = self.transformer(emb)
        feat, distill_tokens = x[:, :-1], x[:, -1]

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits, distill_tokens

# knowledge distillation wrapper
class DistillWrapper(nn.Module):
    def __init__(self, *, teacher, student, temperature=1.0, alpha=0.5):
        super().__init__()
        assert isinstance(student, DistillableVisionTransformer), 'student must be a distillable vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, **kwargs):
        b, *_ = img.shape

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, distill_token=self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        distill_loss = F.kl_div(
            F.log_softmax(distill_logits / self.temperature, dim = -1),
            F.softmax(teacher_logits / self.temperature, dim = -1).detach(),
        reduction = 'batchmean')

        distill_loss *= self.temperature ** 2

        return loss * self.alpha + distill_loss * (1 - self.alpha)
