import os
import glob
from tqdm import tqdm
import cv2
from insightface.app import FaceAnalysis

# LFW图片根目录
lfw_root = '/home/yu/scikit_learn_data/lfw_home/lfw_funneled'
# 输出landmarks文件
landmark_file = os.path.join(lfw_root, '../lfw_landmarks_insightface.txt')

# 初始化insightface
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0)

# batch size for GPU processing
batch_size = 32  # 可根据显存调整
# 遍历所有图片
# img_list = glob.glob(os.path.join(lfw_root, '*', '*.jpg'))
# with open(landmark_file, 'w') as fout:
#     for i in tqdm(range(0, len(img_list), batch_size)):
#         batch_paths = img_list[i:i+batch_size]
#         imgs = [cv2.imread(p) for p in batch_paths]
#         # 过滤掉读取失败的图片
#         valid = [(img, p) for img, p in zip(imgs, batch_paths) if img is not None]
#         if not valid:
#             continue
#         imgs_valid, paths_valid = zip(*valid)
#         for img, img_path in zip(imgs_valid, paths_valid):
#             # img = img / 127.5 - 1.0  # 归一化到[-1, 1]
#             faces = face_app.get(img)
#             if len(faces) == 0:
#                 print(f'No face: {img_path}')
#                 continue
            # landmark = faces[0].kps  # shape: (5,2)
            # pts = [f'{int(x)} {int(y)}' for x, y in landmark]
            # name = os.path.splitext(os.path.basename(img_path))[0]
            # person = os.path.basename(os.path.dirname(img_path))
            # fout.write(f'{person}_{name[-4:]} {" ".join(pts)}\n')
# print(f'Landmarks saved to {landmark_file}')
# /home/yu/celeba/img_align_celeba/000001.jpg
import PIL.Image
import numpy as np
image = PIL.Image.open('/home/yu/celeba/img_align_celeba/000001.jpg').resize((112, 112))
image = np.array(image)
if image.ndim == 2:
    image = image[:, :, np.newaxis]
# image = image.transpose(2, 0, 1)  # HWC -> CHW
# image = np.transpose(image, (1,2,0))
image = image / 127.5 - 1.0  # 归一化到[-1, 1]
image = (image + 1) / 2
image = (image * 255).astype('uint8')  # 转换回[0, 255]范围
# 过滤掉读取失败的图片
faces = face_app.get(image)
print(faces)