import numpy as np
import os

# 特征文件路径
NPZ_PATH = '/home/yu/celeba/celeba_arcface_features_align.npz'
TRAIN_NPZ = '/home/yu/celeba/celeba_arcface_features_align_train.npz'
VAL_NPZ = '/home/yu/celeba/celeba_arcface_features_align_val.npz'

# 读取原始特征
data = np.load(NPZ_PATH, allow_pickle=True)
features = data['features']
img_names = data['img_names']

# 读取已划分的训练/验证图片名
train_dir = '/home/yu/celeba/img_align_celeba_112_train'
val_dir = '/home/yu/celeba/img_align_celeba_112_val'
train_imgs = set(os.listdir(train_dir))
val_imgs = set(os.listdir(val_dir))

# 按图片名划分特征（确保类型一致，全部转为str再比对）
train_imgs_str = set(str(x) for x in train_imgs)
val_imgs_str = set(str(x) for x in val_imgs)
img_names_str = np.array([str(x) for x in img_names])

train_idx = [i for i, n in enumerate(img_names_str) if n in train_imgs_str]
val_idx = [i for i, n in enumerate(img_names_str) if n in val_imgs_str]

train_features = features[train_idx]
train_img_names = img_names[train_idx]
val_features = features[val_idx]
val_img_names = img_names[val_idx]

np.savez(TRAIN_NPZ, features=train_features, img_names=train_img_names)
np.savez(VAL_NPZ, features=val_features, img_names=val_img_names)
print(f"已保存训练特征到 {TRAIN_NPZ}，shape={train_features.shape}")
print(f"已保存验证特征到 {VAL_NPZ}，shape={val_features.shape}")
