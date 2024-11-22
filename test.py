import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS

# CIFAR-10 类别标签
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Prepare Model
config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, num_classes=10, zero_head=False, img_size=224, vis=True)  # vis=True，可视化注意力
model.load_state_dict(torch.load('output/cifar10-100_500_checkpoint.bin'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
img = Image.open('test_img/dog.png')
x = transform(img)

logits, att_mat = model(x.unsqueeze(0))  # logits: 分类的输出分数 att_mat: 模型每一层的注意力矩阵
# 打印预测结果
probs = torch.nn.Softmax(dim=-1)(logits)
top5 = torch.argsort(probs, dim=-1, descending=True)
print("Prediction Label and Attention Map!\n")
for idx in top5[0, :3]:
    print(f'{probs[0, idx.item()]:.5f} : {classes[idx.item()]}')
att_mat = torch.stack(att_mat).squeeze(1)

# 可视化注意力矩阵
# Average the attention weights across all heads.
att_mat = torch.mean(att_mat, dim=1)

# To account for residual connections, we add an identity matrix to the
# attention matrix and re-normalize the weights.
residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

# Attention from the output token to the input space.
v = joint_attentions[-1]
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
# mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
# mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
# result = (mask * img).astype("uint8")

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
#
# ax1.set_title('Original')
# ax2.set_title('Attention Map')
# _ = ax1.imshow(img)
# _ = ax2.imshow(result)

for i, v in enumerate(joint_attentions):
    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    result = (mask * img).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map_%d Layer' % (i+1))
    _ = ax1.imshow(img)
    _ = ax2.imshow(result)
    plt.show()
