import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

sys.path.append("/home/yu/workspace/mia/arcface")


class CelebAImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, landmark_dict=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.img_names.sort()
        self.landmark_dict = landmark_dict
        # 对齐函数
        if landmark_dict is not None:
            import cv2
            from skimage import transform as sk_transform
            import numpy as np
            from PIL import Image
            def face_align(img, landmark, image_size=(112, 112)):
                src = np.array(
                    [[38.2946, 51.6963],
                     [73.5318, 51.5014],
                     [56.0252, 71.7366],
                     [41.5493, 92.3655],
                     [70.7299, 92.2041]], dtype=np.float32)
                tform = sk_transform.SimilarityTransform()
                tform.estimate(landmark, src)
                M = tform.params[0:2, :]
                aligned = cv2.warpAffine(np.array(img), M, image_size, borderValue=0.0)
                return Image.fromarray(aligned)
            self.face_align = face_align
        else:
            self.face_align = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # 对齐
        if self.landmark_dict is not None and img_name in self.landmark_dict:
            image = self.face_align(image, self.landmark_dict[img_name])
        if self.transform:
            image = self.transform(image)
        return image, img_name


class LFWFeatureExtractor:
    """
    用于批量提取LFW图片特征并保存到npz文件。
    用法示例：
        extractor = LFWFeatureExtractor(
            lfw_dir='/path/to/lfw',
            arcface_weight='/path/to/arcface_r50.pth',
            output_path='/path/to/lfw_arcface_features.npz',
            batch_size=64
        )
        extractor.extract_and_save()
    参数：
        lfw_dir: LFW图片文件夹（每个人一个子文件夹，标准LFW格式）
        arcface_weight: ArcFace模型权重路径
        output_path: 特征保存路径（.npz）
        batch_size: DataLoader批量大小
    """

    def __init__(
        self, lfw_dir, arcface_weight, output_path, batch_size=64, num_workers=4
    ):
        self.lfw_dir = lfw_dir
        self.arcface_weight = arcface_weight
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def extract_and_save(self):
        from torchvision import transforms
        from torch.utils.data import DataLoader
        import torch
        import numpy as np
        from tqdm import tqdm
        from backbones import get_model
        import os
        from PIL import Image

        # 构建图片路径列表和图片名列表
        img_paths = []
        img_names = []
        for person in sorted(os.listdir(self.lfw_dir)):
            person_dir = os.path.join(self.lfw_dir, person)
            if not os.path.isdir(person_dir):
                continue
            for fname in sorted(os.listdir(person_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_paths.append(os.path.join(person_dir, fname))
                    img_names.append(f"{person}/{fname}")

        class LFWImageListDataset(Dataset):
            def __init__(self, img_paths, transform=None):
                self.img_paths = img_paths
                self.transform = transform

            def __len__(self):
                return len(self.img_paths)

            def __getitem__(self, idx):
                image = Image.open(self.img_paths[idx]).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image

        transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = LFWImageListDataset(img_paths, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # 加载ArcFace模型
        state_dict = torch.load(self.arcface_weight, map_location="cpu")
        model = get_model("r50")
        if any(k.startswith("module.") for k in state_dict.keys()):
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        features = []
        with torch.no_grad():
            for batch_imgs in tqdm(dataloader, desc="Extracting LFW features"):
                imgs = batch_imgs.cuda() if torch.cuda.is_available() else batch_imgs
                feats = model(imgs).cpu().numpy()
                feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
                features.append(feats)
        features = np.concatenate(features, axis=0)
        np.savez(self.output_path, features=features, img_names=img_names)
        print(f"已保存LFW特征到 {self.output_path}, shape={features.shape}")


# 用法示例：
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
# dataset = CelebAImageDataset('/home/yu/celeba/img_align_celeba', transform=transform)
# image = dataset[0]
# dataset = CelebAFeatureDataset('/home/yu/celeba/img_align_celeba', '/home/yu/celeba/celeba_arcface_features.npz', transform=transform)
# image, feature = dataset[0]
# print(image.shape, feature.shape)
def celeba_arcface_feature_extractor():
    from torchvision import transforms
    import numpy as np
    import torch
    from tqdm import tqdm
    from backbones import get_model
    import cv2
    from skimage import transform as sk_transform
    from PIL import Image
    import os

    # 读取landmarks（如有）
    landmark_file = '/home/yu/celeba/list_landmarks_align_celeba.txt'
    landmark_dict = {}
    if os.path.exists(landmark_file):
        with open(landmark_file, 'r') as f:
            for line in f:
                arr = line.strip().split()
                if len(arr) == 11:
                    name = arr[0]
                    pts = np.array(list(map(float, arr[1:])), dtype=np.float32).reshape(5, 2)
                    landmark_dict[name] = pts

    def face_align(img, landmark, image_size=(112, 112)):
        src = np.array(
            [[38.2946, 51.6963],
             [73.5318, 51.5014],
             [56.0252, 71.7366],
             [41.5493, 92.3655],
             [70.7299, 92.2041]], dtype=np.float32)
        tform = sk_transform.SimilarityTransform()
        tform.estimate(landmark, src)
        M = tform.params[0:2, :]
        aligned = cv2.warpAffine(np.array(img), M, image_size, borderValue=0.0)
        return Image.fromarray(aligned)

    # 加载arcface模型
    state_dict = torch.load(
        "/home/yu/workspace/mia/ckpts/arcface_r100.pth", map_location="cpu"
    )
    model = get_model("r100")
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # 预处理
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = CelebAImageDataset(
        "/home/yu/celeba/img_align_celeba", transform=transform, landmark_dict=landmark_dict if landmark_dict else None
    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    features = []
    img_names = []
    with torch.no_grad():
        for batch_imgs, batch_names in tqdm(dataloader, desc="Extracting CelebA features"):
            imgs = batch_imgs.cuda() if torch.cuda.is_available() else batch_imgs
            feats = model(imgs).cpu().numpy()
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            features.append(feats)
            img_names.extend(batch_names)
    features = np.concatenate(features, axis=0)
    np.savez(
        "/home/yu/celeba/celeba_arcface_features_align.npz",
        features=features,
        img_names=img_names,
    )
    print(
        f"已保存特征到 /home/yu/celeba/celeba_arcface_features_align.npz, shape={features.shape}"
    )


def save_aligned_celeba_images(
    img_dir="/home/yu/celeba/img_align_celeba",
    landmark_file="/home/yu/celeba/list_landmarks_align_celeba.txt",
    out_dir="/home/yu/celeba/img_align_celeba_112"
):
    """
    读取CelebA图片和landmark，对齐后保存到新目录。
    """
    import os
    import numpy as np
    from PIL import Image
    import cv2
    from skimage import transform as sk_transform
    os.makedirs(out_dir, exist_ok=True)
    # 读取landmark
    landmark_dict = {}
    if os.path.exists(landmark_file):
        with open(landmark_file, 'r') as f:
            for line in f:
                arr = line.strip().split()
                if len(arr) == 11:
                    name = arr[0]
                    pts = np.array(list(map(float, arr[1:])), dtype=np.float32).reshape(5, 2)
                    landmark_dict[name] = pts
    img_names = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    img_names.sort()
    src = np.array(
        [[38.2946, 51.6963],
         [73.5318, 51.5014],
         [56.0252, 71.7366],
         [41.5493, 92.3655],
         [70.7299, 92.2041]], dtype=np.float32)
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        out_path = os.path.join(out_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if img_name in landmark_dict:
            landmark = landmark_dict[img_name]
            tform = sk_transform.SimilarityTransform()
            tform.estimate(landmark, src)
            M = tform.params[0:2, :]
            aligned = cv2.warpAffine(np.array(img), M, (112, 112), borderValue=0.0)
            aligned_img = Image.fromarray(aligned)
        else:
            aligned_img = img.resize((112, 112))
        aligned_img.save(out_path)
    print(f"已保存对齐后图片到 {out_dir}")


if __name__ == "__main__":
    # 示例：提取LFW特征
    # extractor = LFWFeatureExtractor(
    #     lfw_dir="/home/yu/lfw/lfw_funneled",
    #     arcface_weight="/home/yu/workspace/mia/ckpts/arcface_r50.pth",
    #     output_path="/home/yu/workspace/mia/data/lfw_arcface_features.npz",
    #     batch_size=64,
    # )
    # extractor.extract_and_save()

    # # 示例：提取CelebA特征
    # celeba_arcface_feature_extractor()
    # 生成对齐后CelebA图片
    save_aligned_celeba_images()
