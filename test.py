import torch
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam

student = DistillableVisionTransformer.from_pretrained('ViT-B_16', weights_path='weights/ViT-B_16.pth', num_classes=5).cuda()
teacher = EfficientNet.from_name('efficientnet-b0', num_classes=5)
checkpoint = torch.load('save/simple.pt')
teacher.load_state_dict(checkpoint['model'])
teacher.cuda()

distiller = DistillWrapper(
        student=student,
        teacher=teacher,
        temperature=3,
        alpha=0.5
).cuda()

optimizer = Adam(distiller.parameters(), lr=1e-3)

distiller.eval()

inputs = torch.randn(1, 3, *student.image_size).cuda()
labels = torch.zeros([1], dtype=torch.long).cuda()
print(distiller(inputs, labels).item())

