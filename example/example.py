import torch
from torch.optim import Adam
from distillable_vision_transformer import DistillableVisionTransformer, DistillationTrainer, DistilledLoss
from efficientnet_pytorch import EfficientNet

inputs = torch.rand(32, 3, 384, 384).cuda()
labels = torch.zeros(32).long().cuda()

student = DistillableVisionTransformer.from_pretrained('ViT-B_16').cuda()
teacher = EfficientNet.from_pretrained('efficientnet-b0').cuda()

trainer = DistillationTrainer(teacher=teacher, student=student).cuda()
optimizer = Adam(trainer.parameters(), lr=1e-3)
trainer.train()
loss = DistilledLoss(alpha=0.5, temperature=1.0)

optimizer.zero_grad()
teacher_logits, student_logits, distill_logits = trainer(inputs)
l = loss(teacher_logits, student_logits, distill_logits, labels)
l.backward()
optimizer.step()

print('loss', l.item())
