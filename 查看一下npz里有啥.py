import numpy as np

npz_file = np.load('checkpoint/ViT-B_16.npz')
# print(npz_file.files)  # 查看所有的键（即数组的名称）

# 查看某个数组
array = npz_file['Transformer/encoder_norm/bias']  # 替换为你需要查看的数组名
print(array)
