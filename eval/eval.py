import os
import csv
import random
from PIL import Image, ImageDraw, ImageFont

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from torchvision.utils import save_image
import sys

sys.path.append("/home/yu/workspace/mia/edm")
sys.path.append("/home/yu/workspace/mia/arcface")
import dnnlib
import pickle
from tqdm import tqdm
import numpy as np
import PIL.Image
import cv2
from insightface.app import FaceAnalysis
from skimage.transform import SimilarityTransform


def load_model(network_pkl, device):
    print(f"Loading model from {network_pkl}...")
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device)
    net.eval()
    return net


def generate_images(
    net,
    num_images=16,
    label_vectors=None,
    device="cuda",
    outdir="./eval_results",
    img_names=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    """
    仿照edm/example.py的采样流程，支持512维特征条件输入。
    """
    import math

    os.makedirs(outdir, exist_ok=True)
    batch_size_default = 8  # 逐batch采样，节省显存
    img_resolution = getattr(net, "img_resolution", 128)
    img_channels = getattr(net, "img_channels", 3)
    total = num_images
    generated = 0
    with torch.no_grad():
        pbar = tqdm(total=total, desc="Sampling", unit="img")
        while generated < total:
            batch_size = min(batch_size_default, total - generated)
            # 采样噪声
            x_next = torch.randn(
                [batch_size, img_channels, img_resolution, img_resolution],
                device=device,
            )  # [B,3,H,W]
            # label
            if label_vectors is not None:
                labels = label_vectors[generated : generated + batch_size]
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            else:
                labels = None
            # 步进参数
            sigma_min_ = max(sigma_min, net.sigma_min)
            sigma_max_ = min(sigma_max, net.sigma_max)
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
            t_steps = (
                sigma_max_ ** (1 / rho)
                + step_indices
                / (num_steps - 1)
                * (sigma_min_ ** (1 / rho) - sigma_max_ ** (1 / rho))
            ) ** rho
            t_steps = torch.cat(
                [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
            )  # t_N = 0
            x_next = x_next.to(torch.float64) * t_steps[0]
            # 主采样循环
            for i, (t_cur, t_next_) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_cur = x_next
                # S_churn: 临时加噪
                gamma = (
                    min(S_churn / num_steps, math.sqrt(2) - 1)
                    if S_min <= t_cur <= S_max
                    else 0
                )
                t_hat = net.round_sigma(t_cur + gamma * t_cur)
                x_hat = x_cur + (
                    t_hat**2 - t_cur**2
                ).sqrt() * S_noise * torch.randn_like(x_cur)
                # Euler step
                denoised = net(x_hat, t_hat, labels).to(torch.float64)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next_ - t_hat) * d_cur
                # 2阶修正
                if i < num_steps - 1:
                    denoised = net(x_next, t_next_, labels).to(torch.float64)
                    d_prime = (x_next - denoised) / t_next_
                    x_next = x_hat + (t_next_ - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                del x_cur, x_hat, denoised, d_cur
            # 保存图片
            images = (x_next * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            for i in range(images.shape[0]):
                if img_names is not None:
                    name = (
                        img_names[generated + i]
                        .replace("/", "_")
                        .replace(".jpg", ".png")
                        .replace(".jpeg", ".png")
                        .replace(".png", ".png")
                    )
                else:
                    name = f"gen_{generated+i:05d}.png"
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                PIL.Image.fromarray(img, "RGB").save(os.path.join(outdir, name))
            generated += batch_size
            pbar.update(batch_size)
        pbar.close()


def eval_lfw():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 修改为你的模型路径
    network_pkl = "/home/yu/workspace/mia/training-runs/00046-celeba_feature-cond-ddpmpp-edm-gpus4-batch64-fp32/network-snapshot-019018.pkl"
    net = load_model(network_pkl, device)
    # 加载 LFW 特征作为条件标签，生成对应图片
    lfw_feature_path = "/home/yu/workspace/mia/data/lfw_arcface_features.npz"
    max_num = 100  # 限制生成图片数量
    if os.path.exists(lfw_feature_path):
        lfw_data = np.load(lfw_feature_path, allow_pickle=True)
        label_vectors = lfw_data["features"]  # shape: [num_images, 512]
        img_names = lfw_data["img_names"] if "img_names" in lfw_data else None
        num_to_generate = min(max_num, label_vectors.shape[0])
        print(
            f"加载 LFW 特征 shape={label_vectors.shape}，开始生成前{num_to_generate}张图片..."
        )
        generate_images(
            net,
            num_images=num_to_generate,
            label_vectors=label_vectors[:num_to_generate],
            device=device,
            outdir="./eval_results_lfw_46",
            img_names=img_names[:num_to_generate],
        )
        print("图片已保存到 ./eval_results_lfw_04")
    else:
        print(f"未找到特征文件: {lfw_feature_path}")


def eval_face_recognition_lfw(batch_size=64):
    """
    用ArcFace特征做人脸识别，测试生成图片被识别为原人的Top-1准确率。
    以eval_results_lfw下实际生成的图片为准，数据库每个人保存所有特征，检索时取最大相似度。
    需要 lfw_arcface_features.npz 里有 img_names（如 xxxx.jpg），并有 pairs.txt 或 identity 映射。
    batch_size: 生成图片特征提取时的批量大小。
    """
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    import os

    sys.path.append("/home/yu/workspace/mia/arcface")
    from backbones import get_model

    # 路径
    gen_dir = "./eval_results_lfw_46"
    feature_path = "/home/yu/workspace/mia/data/lfw_arcface_features.npz"
    arcface_weight = "/home/yu/workspace/mia/ckpts/arcface_r100.pth"

    # 加载原始特征
    data = np.load(feature_path, allow_pickle=True)
    orig_features = data["features"]
    img_names = data["img_names"]

    # identity: 用文件名前缀（如Abe_Lincoln_0001.jpg -> Abe_Lincoln）聚合
    def get_identity(name):
        return os.path.basename(name).rsplit("_", 1)[0]

    pid2feats = {}
    for feat, name in zip(orig_features, img_names):
        pid = get_identity(name)
        pid2feats.setdefault(pid, []).append(feat)
    pid_list = sorted(pid2feats.keys())

    # 获取生成图片名列表
    gen_img_files = [
        f for f in os.listdir(gen_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    gen_img_files.sort()
    # 反查person_id
    gen_img2pid = {f: get_identity(f.replace(".png", ".jpg")) for f in gen_img_files}
    valid_pairs = [(f, pid) for f, pid in gen_img2pid.items() if pid in pid2feats]
    if len(valid_pairs) == 0:
        print("未找到有效的图片与person_id对应关系")
        return

    # 加载ArcFace模型
    state_dict = torch.load(arcface_weight, map_location="cpu")
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

    # 批量提取生成图片特征
    gen_features = []
    gt_pid = []
    n = len(valid_pairs)
    for i in tqdm(range(0, n, batch_size), desc="Extract gen features (batch)"):
        batch_pairs = valid_pairs[i : i + batch_size]
        batch_imgs = []
        batch_pids = []
        for f, pid in batch_pairs:
            gen_path = os.path.join(gen_dir, f)
            img = Image.open(gen_path).convert("RGB")
            # 人脸对齐
            aligned = align_face_with_retinaface(img, face_app)
            img_tensor = transform(Image.fromarray(aligned))
            batch_imgs.append(img_tensor)
            batch_pids.append(pid)
        batch_imgs = torch.stack(batch_imgs, dim=0)
        batch_imgs = batch_imgs.cuda() if torch.cuda.is_available() else batch_imgs
        with torch.no_grad():
            feats = model(batch_imgs).cpu().numpy()
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        gen_features.append(feats)
        gt_pid.extend(batch_pids)
    gen_features = np.concatenate(gen_features, axis=0)  # [N, 512]
    gt_pid = np.array(gt_pid)

    # --- GPU并行检索优化 ---
    # 拼接所有identity的特征为大tensor，记录每个人的起止索引
    all_feats = []
    pid_ptr = [0]
    for pid in pid_list:
        feats_this = np.stack(pid2feats[pid], axis=0)
        all_feats.append(feats_this)
        pid_ptr.append(pid_ptr[-1] + feats_this.shape[0])
    all_feats = np.concatenate(all_feats, axis=0)  # [M, 512]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_feats = torch.tensor(all_feats, dtype=torch.float32, device=device)
    all_feats = all_feats / all_feats.norm(dim=1, keepdim=True)
    gen_features_torch = torch.tensor(gen_features, dtype=torch.float32, device=device)
    gen_features_torch = gen_features_torch / gen_features_torch.norm(
        dim=1, keepdim=True
    )

    # 并行计算余弦相似度 [N, M]，分段max
    with torch.no_grad():
        sim_matrix = torch.matmul(gen_features_torch, all_feats.T)  # [N, M]
        # 向量化分段max：每个identity对应一段，拼成[N, num_identity]
        sim_per_identity = torch.stack(
            [
                sim_matrix[:, pid_ptr[j] : pid_ptr[j + 1]].max(dim=1).values
                for j in range(len(pid_list))
            ],
            dim=1,
        )  # [N, num_identity]
        max_sim, pred_idx = sim_per_identity.max(dim=1)  # [N]
        pred_pid_list = [pid_list[i] for i in pred_idx.cpu().numpy()]
        max_sim_list = max_sim.cpu().numpy().tolist()
        correct = (np.array(pred_pid_list) == gt_pid).sum()
        acc = correct / len(gen_features)
    print(
        f"LFW人脸识别Top-1准确率(按identity, 全特征库): {acc*100:.2f}% (共{len(gen_features)}张有效图片)"
    )
    # 保存每个生成图片的最大相似度到文件
    out_path = os.path.join(gen_dir, "face_recognition_max_sim.csv")
    with open(out_path, "w") as f:
        f.write("filename,gt_pid,pred_pid,max_sim\n")
        for (fimg, pid), pred, sim in zip(valid_pairs, pred_pid_list, max_sim_list):
            f.write(f"{fimg},{pid},{pred},{sim:.6f}\n")
    print(f"每个生成图片的最大相似度已保存到: {out_path}")
    return


def extract_lfw_arcface_features(batch_size=64):
    """
    提取LFW图片的ArcFace特征，保存为npz文件，支持batch处理。
    输出文件：/home/yu/workspace/mia/data/lfw_arcface_features.npz
    """
    import os
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    import torch
    from torchvision import transforms

    sys.path.append("/home/yu/workspace/mia/arcface")
    from backbones import get_model

    lfw_dir = "/home/yu/scikit_learn_data/lfw_home/lfw_funneled"
    out_path = "/home/yu/workspace/mia/data/lfw_arcface_features.npz"
    arcface_weight = "/home/yu/workspace/mia/ckpts/arcface_r50.pth"

    img_paths = []
    img_names = []
    for root, dirs, files in os.walk(lfw_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(root, f))
                person = os.path.basename(root)
                img_names.append(f"{person}_{f}")
    print(f"共收集到图片: {len(img_paths)}")
    if len(img_paths) == 0:
        raise RuntimeError(f"未在 {lfw_dir} 下找到任何图片，请检查路径和数据集结构！")

    # 加载ArcFace模型
    state_dict = torch.load(arcface_weight, map_location="cpu")
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

    # 预处理
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    features = []
    n = len(img_paths)
    for i in tqdm(range(0, n, batch_size), desc="Extract ArcFace features"):
        batch_paths = img_paths[i : i + batch_size]
        batch_imgs = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            aligned = align_face_with_retinaface(img, face_app)
            img_tensor = transform(Image.fromarray(aligned))
            batch_imgs.append(img_tensor)
        batch_imgs = torch.stack(batch_imgs, dim=0)
        batch_imgs = batch_imgs.cuda() if torch.cuda.is_available() else batch_imgs
        with torch.no_grad():
            feats = model(batch_imgs).cpu().numpy()
            feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        features.append(feats)
    features = np.concatenate(features, axis=0)
    img_names_np = np.array(img_names)
    np.savez(out_path, features=features, img_names=img_names_np)
    print(f"LFW ArcFace特征已保存到: {out_path}，shape={features.shape}")


def generate_and_eval_subset_celeba(
    num=1000,
    batch_size=32,
    out_dir="./outputs/eval_results_celeba_subset_1000_00046_16015",
    sim_threshold=0.5,
):
    """
    从celeba特征库随机选num个特征，按identity聚类，生成图片并仅在这num个特征中做相似度评测。
    计算Top-1和Top-5准确率。最大相似度低于sim_threshold时视为未匹配。
    """
    import random
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm

    sys.path.append("/home/yu/workspace/mia/arcface")
    from backbones import get_model

    # 路径
    feature_path = "/home/yu/celeba/celeba_arcface_features_align_val.npz"
    arcface_weight = "/home/yu/workspace/mia/ckpts/arcface_r100.pth"
    identity_file = "/home/yu/celeba/identity_CelebA.txt"
    network_pkl = "/home/yu/workspace/mia/training-runs/00046-celeba_feature-cond-ddpmpp-edm-gpus4-batch64-fp32/network-snapshot-016015.pkl"

    # 加载identity映射
    img2pid = {}
    with open(identity_file, "r") as f:
        for line in f:
            img, pid = line.strip().split()
            img2pid[img] = int(pid)

    # 加载特征库
    data = np.load(feature_path, allow_pickle=True)
    orig_features = data["features"]
    img_names = data["img_names"]
    # 过滤有identity的图片
    valid = [
        (feat, name, img2pid.get(os.path.basename(name), None))
        for feat, name in zip(orig_features, img_names)
    ]
    valid = [x for x in valid if x[2] is not None]
    # 随机采样num个，确保图片名唯一
    sampled = random.sample(valid, num)
    feats, names, pids = zip(*sampled)
    feats = np.stack(feats, axis=0)
    pids = list(pids)
    # 生成唯一图片名，防止覆盖
    gen_img_files = [f"gen_{i:06d}.png" for i in range(num)]
    # identity分组
    pid2feats = {}
    for feat, pid in zip(feats, pids):
        pid2feats.setdefault(pid, []).append(feat)
    pid_list = sorted(pid2feats.keys())

    # 生成图片
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(network_pkl, device)
    os.makedirs(out_dir, exist_ok=True)
    # 生成图片名与原图名一一对应（去除路径，仅保留文件名，后缀统一为.png）
    gen_img_files = [os.path.splitext(os.path.basename(n))[0] + ".png" for n in names]
    generate_images(
        net,
        num_images=num,
        label_vectors=feats,
        device=device,
        outdir=out_dir,
        img_names=gen_img_files,
    )

    # ArcFace模型
    state_dict = torch.load(arcface_weight, map_location="cpu")
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
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # 批量提取生成图片特征
    gen_features = []
    gt_pid = []
    valid_pairs = []
    face_detected = 0
    total_imgs = 0
    for i, (f, pid) in enumerate(zip(gen_img_files, pids)):
        gen_path = os.path.join(out_dir, f)
        if not os.path.exists(gen_path):
            print(f"[警告] 生成图片 {gen_path} 不存在，跳过")
            continue
        try:
            img = Image.open(gen_path).convert("RGB")
        except Exception as e:
            print(f"[警告] 生成图片 {gen_path} 读取失败: {e}")
            continue
        # 人脸对齐
        aligned = align_face_with_retinaface(img, face_app)
        # debug: 检查是否检测到人脸
        faces = face_app.get(np.array(img))
        total_imgs += 1
        if len(faces) > 0:
            face_detected += 1
        else:
            print(f"[DEBUG] 未检测到人脸: {gen_path}")
        img_tensor = transform(Image.fromarray(aligned))
        gen_features.append(img_tensor)
        gt_pid.append(pid)
        valid_pairs.append((f, pid))
    print(
        f"[DEBUG] 生成图片总数: {total_imgs}, 检测到人脸数: {face_detected}, 检测率: {face_detected/total_imgs if total_imgs>0 else 0:.4f}"
    )
    if len(gen_features) == 0:
        print("[错误] 没有成功提取任何生成图片特征，流程终止！")
        return
    gen_features = torch.stack(gen_features, dim=0)
    gen_features = gen_features.cuda() if torch.cuda.is_available() else gen_features
    with torch.no_grad():
        feats = model(gen_features).cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    print(
        f"[DEBUG] gen_features shape: {gen_features.shape}, feats shape: {feats.shape}"
    )
    print(f"[DEBUG] feats mean: {feats.mean():.4f}, std: {feats.std():.4f}")
    gt_pid = np.array(gt_pid)

    # --- 验证过程与eval_face_recognition_celeba一致 ---
    # 拼接所有identity的特征为大tensor，记录每个人的起止索引
    all_feats = []
    pid_ptr = [0]
    for pid in pid_list:
        feats_this = np.stack(pid2feats[pid], axis=0)
        all_feats.append(feats_this)
        pid_ptr.append(pid_ptr[-1] + feats_this.shape[0])
    all_feats = np.concatenate(all_feats, axis=0)  # [M, 512]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_feats = torch.tensor(all_feats, dtype=torch.float32, device=device)
    all_feats = all_feats / all_feats.norm(dim=1, keepdim=True)
    gen_features_torch = torch.tensor(feats, dtype=torch.float32, device=device)
    gen_features_torch = gen_features_torch / gen_features_torch.norm(
        dim=1, keepdim=True
    )

    # 并行计算余弦相似度 [N, M]，分段max
    with torch.no_grad():
        sim_matrix = torch.matmul(gen_features_torch, all_feats.T)  # [N, M]
        # 向量化分段max：每个identity对应一段，拼成[N, num_identity]
        sim_per_identity = torch.stack(
            [
                sim_matrix[:, pid_ptr[j] : pid_ptr[j + 1]].max(dim=1).values
                for j in range(len(pid_list))
            ],
            dim=1,
        )  # [N, num_identity]
        max_sim, pred_idx = sim_per_identity.max(dim=1)  # [N]
        # 阈值处理：低于阈值视为未匹配
        pred_pid_list = []
        for i, (sim, idx) in enumerate(
            zip(max_sim.cpu().numpy(), pred_idx.cpu().numpy())
        ):
            if sim < sim_threshold:
                pred_pid_list.append(-1)  # -1表示未匹配
            else:
                pred_pid_list.append(pid_list[idx])
        max_sim_list = max_sim.cpu().numpy().tolist()
        correct = (
            (np.array(pred_pid_list) == gt_pid) & (np.array(pred_pid_list) != -1)
        ).sum()
        acc = correct / len(gen_features)
        # Top-5
        top5_idx = (
            torch.topk(sim_per_identity, k=min(5, len(pid_list)), dim=1)
            .indices.cpu()
            .numpy()
        )  # [N,5]
        top5_pid = np.array(pid_list)[top5_idx]  # [N,5]
        gt_pid_arr = np.array(gt_pid).reshape(-1, 1)
        # Top-5只统计有匹配的
        top5_correct = (
            (
                (top5_pid == gt_pid_arr)
                & (max_sim[:, None].cpu().numpy() >= sim_threshold)
            )
            .any(axis=1)
            .sum()
        )
        acc_top5 = top5_correct / len(gen_features)
    print(
        f"[Subset] 人脸识别Top-1准确率: {acc*100:.2f}% (共{len(gen_features)}张, 阈值={sim_threshold})"
    )
    print(
        f"[Subset] 人脸识别Top-5准确率: {acc_top5*100:.2f}% (共{len(gen_features)}张, 阈值={sim_threshold})"
    )
    # 保存csv
    out_path = os.path.join(out_dir, "face_recognition_max_sim.csv")
    with open(out_path, "w") as f:
        f.write("filename,gt_pid,pred_pid,max_sim,top5_pred_pids\n")
        for (fimg, pid), pred, sim, top5 in zip(
            valid_pairs, pred_pid_list, max_sim_list, top5_pid
        ):
            top5_str = "|".join(str(x) for x in top5)
            f.write(f"{fimg},{pid},{pred},{sim:.6f},{top5_str}\n")
    print(f"[Subset] 每个生成图片的最大相似度已保存到: {out_path}")
    return


def select_by_lpips(gen_dir, real_img_root, gen_imgs, n_select, img_size=(160,160)):
    """
    计算生成图片与原图的LPIPS感知损失，返回损失最小的n_select个索引
    """
    import lpips
    import torch
    from PIL import Image
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    scores = []
    for idx, gen_name in enumerate(gen_imgs):
        base = os.path.splitext(gen_name)[0]
        real_img_name = base + '.jpg'
        real_path = os.path.join(real_img_root, real_img_name)
        if not os.path.exists(real_path):
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(real_img_root, base + ext)
                if os.path.exists(alt_path):
                    real_path = alt_path
                    break
        gen_path = os.path.join(gen_dir, gen_name)
        if not (os.path.exists(gen_path) and os.path.exists(real_path)):
            scores.append((float('inf'), idx))
            continue
        # 加载图片并resize
        def pil2tensor(img):
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr.transpose(2,0,1)  # CHW
            return torch.from_numpy(arr).unsqueeze(0) * 2 - 1  # [-1,1]
        gen_img = Image.open(gen_path).convert('RGB').resize(img_size)
        real_img = Image.open(real_path).convert('RGB').resize(img_size)
        gen_tensor = pil2tensor(gen_img).to(device)
        real_tensor = pil2tensor(real_img).to(device)
        with torch.no_grad():
            score = loss_fn(gen_tensor, real_tensor).item()
        scores.append((score, idx))
    # 取最小的n_select个
    scores.sort()
    selected = [gen_imgs[i] for (_, i) in scores[:n_select]]
    return selected


# 人脸对齐函数（RetinaFace+5点）
def align_face_with_retinaface(img, face_app, image_size=(112, 112)):
    """
    img: PIL.Image or np.ndarray (RGB)
    face_app: insightface.app.FaceAnalysis 实例
    return: 对齐后np.ndarray (RGB)
    """
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    faces = face_app.get(img)
    if len(faces) == 0:
        return cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)
    landmark = faces[0].kps.astype(np.float32)  # (5,2)
    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    tform = SimilarityTransform()
    tform.estimate(landmark, src)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    return aligned


# 初始化insightface
face_app = FaceAnalysis(allowed_modules=["detection", "landmark"])
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)


def print_feature_stats_from_npz(npz_path):
    """
    计算并打印npz特征库(features字段)的均值和方差。
    """
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    print(f"[特征库统计] {npz_path}")
    print(f"shape: {features.shape}")
    print(f"mean: {features.mean():.6f}, std: {features.std():.6f}")
    print(f"min: {features.min():.6f}, max: {features.max():.6f}")
    print(
        f"norm mean: {np.linalg.norm(features, axis=1).mean():.6f}, norm std: {np.linalg.norm(features, axis=1).std():.6f}"
    )


# 用法示例：
# print_feature_stats_from_npz('/home/yu/celeba/celeba_arcface_features_align.npz')
# print_feature_stats_from_npz('/home/yu/workspace/mia/data/lfw_arcface_features.npz')


def visualize_gen_vs_real_face_similarity(
    gen_dir="./eval_results_celeba",
    feature_path="/home/yu/celeba/celeba_arcface_features_align.npz",
    arcface_weight="/home/yu/workspace/mia/ckpts/arcface_r100.pth",
    identity_file="/home/yu/celeba/identity_CelebA.txt",
    out_dir="./visualize_gen_vs_real",
    max_num=100,
):
    """
    对每个生成图片，找到其对应原图，输出：
    - 生成图片
    - 原图片
    - 两者对齐后的人脸图片
    - ArcFace特征相似度
    并保存对比可视化图和csv。
    """
    import os
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    import torch
    import cv2
    from tqdm import tqdm

    sys.path.append("/home/yu/workspace/mia/arcface")
    from backbones import get_model

    os.makedirs(out_dir, exist_ok=True)

    # identity映射
    img2pid = {}
    with open(identity_file, "r") as f:
        for line in f:
            img, pid = line.strip().split()
            img2pid[img] = int(pid)

    # 原始特征库
    data = np.load(feature_path, allow_pickle=True)
    orig_features = data["features"]
    img_names = data["img_names"]
    name2idx = {
        os.path.splitext(os.path.basename(n))[0]: i for i, n in enumerate(img_names)
    }

    # 加载ArcFace
    state_dict = torch.load(arcface_weight, map_location="cpu")
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
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # 初始化insightface
    from insightface.app import FaceAnalysis
    from skimage.transform import SimilarityTransform

    face_app = FaceAnalysis(allowed_modules=["detection", "landmark"])
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

    def align_face(img, face_app, image_size=(112, 112)):
        if isinstance(img, Image.Image):
            img = np.array(img)
        faces = face_app.get(img)
        if len(faces) == 0:
            return cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)
        landmark = faces[0].kps.astype(np.float32)
        src = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        tform = SimilarityTransform()
        tform.estimate(landmark, src)
        M = tform.params[0:2, :]
        aligned = cv2.warpAffine(img, M, image_size, borderValue=0.0)
        return aligned

    # 遍历生成图片
    gen_imgs = [
        f for f in os.listdir(gen_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    gen_imgs.sort()
    rows = []
    for i, gen_img in enumerate(tqdm(gen_imgs[:max_num], desc="Visualize sim")):
        gen_path = os.path.join(gen_dir, gen_img)
        gen_base = os.path.splitext(os.path.basename(gen_img))[0]
        # 找到原图索引
        idx = name2idx.get(gen_base, None)
        if idx is None:
            print(f"[WARN] 找不到原图: {gen_base}")
            continue
        orig_img_path = "/home/yu/celeba/img_align_celeba/" + img_names[idx]
        print(orig_img_path)
        orig_img = (
            Image.open(orig_img_path).convert("RGB")
            if os.path.exists(orig_img_path)
            else None
        )
        if orig_img is None:
            print(f"[WARN] 原图文件不存在: {orig_img_path}")
            continue
        # 读取生成图片
        gen_img_pil = Image.open(gen_path).convert("RGB")
        # 对齐
        aligned_gen = align_face(gen_img_pil, face_app)
        aligned_orig = align_face(orig_img, face_app)
        # ArcFace特征
        with torch.no_grad():
            gen_tensor = transform(Image.fromarray(aligned_gen)).unsqueeze(0)
            orig_tensor = transform(Image.fromarray(aligned_orig)).unsqueeze(0)
            if torch.cuda.is_available():
                gen_tensor = gen_tensor.cuda()
                orig_tensor = orig_tensor.cuda()
            gen_feat = model(gen_tensor).cpu().numpy()[0]
            orig_feat = model(orig_tensor).cpu().numpy()[0]
            gen_feat = gen_feat / np.linalg.norm(gen_feat)
            orig_feat = orig_feat / np.linalg.norm(orig_feat)
            sim = float(np.dot(gen_feat, orig_feat))
        # 保存可视化
        vis_img = np.concatenate(
            [
                np.array(gen_img_pil.resize((112, 112))),
                aligned_gen,
                aligned_orig,
                np.array(orig_img.resize((112, 112))),
            ],
            axis=1,
        )
        vis_path = os.path.join(out_dir, f"{gen_base}_compare.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        rows.append(
            {
                "gen_img": gen_path,
                "orig_img": orig_img_path,
                "aligned_gen": vis_path,
                "sim": sim,
            }
        )
    # 保存csv
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "gen_vs_real_similarity.csv")
    df.to_csv(csv_path, index=False)
    print(f"详细对比结果已保存到: {csv_path}")
    print(f"可视化图片已保存到: {out_dir}")
    return df


def compute_lpips_for_gen_vs_real(
    gen_dir,
    real_dir,
    out_csv_name="gen_vs_real_similarity_with_lpips.csv",
    lpips_key="lpips",
    net="alex",
    topk=20,
    min_lpips_dir="min_lpips_images",
):
    """
    遍历gen_dir下所有图片，与real_dir下同名图片（自动适配后缀）配对，计算LPIPS，保存csv，并拷贝topk最小对及对比图。
    """
    import pandas as pd
    from PIL import Image
    import torchvision.transforms as transforms
    import torch
    import os
    import lpips
    import shutil
    import numpy as np

    # 支持的图片后缀
    exts = [".png", ".jpg", ".jpeg"]
    # 获取gen_dir下所有图片
    gen_imgs = [
        f for f in os.listdir(gen_dir) if os.path.splitext(f)[1].lower() in exts
    ]
    gen_imgs.sort()
    print(f"共检测到生成图片: {len(gen_imgs)}")

    # 初始化lpips模型
    loss_fn = lpips.LPIPS(net=net)
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    def pil_to_tensor(img):
        tf = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        tensor = tf(img) * 2 - 1  # [0,1] -> [-1,1]
        return (
            tensor.unsqueeze(0).cuda()
            if torch.cuda.is_available()
            else tensor.unsqueeze(0)
        )

    rows = []
    for fname in gen_imgs:
        gen_img_path = os.path.join(gen_dir, fname)
        # 匹配real_dir下同名图片（支持不同后缀）
        base = os.path.splitext(fname)[0]
        real_img_path = None
        for ext in exts:
            candidate = os.path.join(real_dir, base + ext)
            if os.path.exists(candidate):
                real_img_path = candidate
                break
        if real_img_path is None:
            print(f"[WARN] 未找到原图: {base} 在 {real_dir}")
            continue
        try:
            img0 = Image.open(gen_img_path).convert("RGB")
            img1 = Image.open(real_img_path).convert("RGB")
            t0 = pil_to_tensor(img0)
            t1 = pil_to_tensor(img1)
            with torch.no_grad():
                score = loss_fn(t0, t1).item()
            rows.append(
                {"gen_img": gen_img_path, "orig_img": real_img_path, lpips_key: score}
            )
        except Exception as e:
            print(f"[WARN] 计算LPIPS失败: {gen_img_path}, {real_img_path}, {e}")
            continue
    if not rows:
        print("[ERROR] 没有成功配对的图片，流程终止！")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(by=lpips_key, ascending=True)
    out_csv_path = os.path.join(gen_dir, out_csv_name)
    df.to_csv(out_csv_path, index=False)
    print(f"已完成LPIPS计算，结果已保存到: {out_csv_path}")

    # --------- 拷贝LPIPS最小的前topk对图片及对比图 ---------
    if topk > 0 and min_lpips_dir:
        min_dir = os.path.join(gen_dir, min_lpips_dir)
        os.makedirs(min_dir, exist_ok=True)
        df_valid = df.head(topk)
        for i, row in df_valid.iterrows():
            gen_img_path = row["gen_img"]
            real_img_path = row["orig_img"]
            lpips_val = row[lpips_key]
            gen_dst = os.path.join(min_dir, f"{i:03d}_lpips{lpips_val:.4f}_gen.png")
            real_dst = os.path.join(min_dir, f"{i:03d}_lpips{lpips_val:.4f}_real.png")
            compare_dst = os.path.join(
                min_dir, f"{i:03d}_lpips{lpips_val:.4f}_compare.png"
            )
            try:
                shutil.copyfile(gen_img_path, gen_dst)
                shutil.copyfile(real_img_path, real_dst)
                # 生成对比图
                gen_img = Image.open(gen_img_path).convert("RGB").resize((112, 112))
                real_img = Image.open(real_img_path).convert("RGB").resize((112, 112))
                compare_img = np.concatenate(
                    [np.array(gen_img), np.array(real_img)], axis=1
                )
                Image.fromarray(compare_img).save(compare_dst)
            except Exception as e:
                print(
                    f"[WARN] 拷贝/生成对比图失败: {gen_img_path}, {real_img_path}, {e}"
                )
        print(f"LPIPS最小的前{topk}对图片及对比图已拷贝到: {min_dir}")
    return


def visualize_gen_vs_real_matrix(
    gen_dir,
    csv_path,
    real_img_root,
    identity_file,
    save_path,
    num_images=16,
    nrow=4,
    ncol=4,
    img_size=(160,160),
    img_gap=20,
    matrix_gap=60,
    font_path=None,
    font_size=28,
    select_mode='lpips'  # 可选 'random' 或 'lpips'
):
    """
    生成图片与原图矩阵可视化，每个小图上方用标题标注类别，矩阵有间距，生成图在左，原图在右。支持自定义字体和字号。
    """
    import os
    import csv
    import random
    from PIL import Image, ImageDraw, ImageFont

    def get_font(font_path, font_size):
        if font_path:
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                return ImageFont.load_default()
        else:
            return ImageFont.load_default()

    # 1. 读取identity文件，建立图片名->id映射
    img2id = {}
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img2id[parts[0]] = parts[1]
    # 2. 读取csv，按你的格式解析
    pred_map = {}
    gt_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                img_name = row[0]
                top1 = row[1]
                gt = row[2]
                pred_map[img_name] = top1
                gt_map[img_name] = gt
    # 3. 获取生成图片列表
    gen_imgs = [
        f for f in os.listdir(gen_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))
    ]
    if select_mode == 'lpips' and len(gen_imgs) > num_images:
        gen_imgs = select_by_lpips(gen_dir, real_img_root, gen_imgs, num_images, img_size)
    elif len(gen_imgs) > num_images:
        import random
        gen_imgs = random.sample(gen_imgs, num_images)
    gen_imgs = sorted(gen_imgs)[:num_images]
    # 4. 查找原图路径和真实id
    real_imgs = []
    real_ids = []
    gen_ids = []
    for gen_name in gen_imgs:
        base = os.path.splitext(gen_name)[0]
        real_img_name = base + ".jpg"
        real_path = os.path.join(real_img_root, real_img_name)
        if not os.path.exists(real_path):
            for ext in [".png", ".jpeg"]:
                alt_path = os.path.join(real_img_root, base + ext)
                if os.path.exists(alt_path):
                    real_path = alt_path
                    break
        real_imgs.append(real_path if os.path.exists(real_path) else None)
        # 优先用csv的gt，否则用identity文件
        real_ids.append(gt_map.get(gen_name, img2id.get(real_img_name, "N/A")))
        gen_ids.append(pred_map.get(gen_name, "N/A"))

    # 5. 加载图片，生成带标题的小图（标题区域加高）
    def load_img_with_title(img_path, title, size=img_size):
        from PIL import Image, ImageDraw, ImageFont

        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB").resize(size)
        else:
            img = Image.new("RGB", size, (128, 128, 128))
        font = get_font(font_path, font_size)
        # 增大标题区域高度
        title_h = int(size[1] * 0.35)
        out = Image.new("RGB", (size[0], size[1] + title_h), (255, 255, 255))
        draw = ImageDraw.Draw(out)
        # 居中写标题，竖直方向也居中
        text_w = draw.textlength(str(title), font=font)
        text_h = font.getbbox(str(title))[3] - font.getbbox(str(title))[1]
        draw.text(
            ((size[0] - text_w) // 2, (title_h - text_h) // 2),
            str(title),
            fill=(0, 0, 0),
            font=font,
        )
        out.paste(img, (0, title_h))
        return out

    gen_matrix = [
        load_img_with_title(os.path.join(gen_dir, n), gen_ids[i], img_size)
        for i, n in enumerate(gen_imgs)
    ]
    real_matrix = [
        load_img_with_title(p, real_ids[i], img_size) for i, p in enumerate(real_imgs)
    ]

    # 6. 拼接为有间距的矩阵
    def make_grid(imgs, nrow, ncol, gap=img_gap):
        if not imgs:
            return None
        w, h = imgs[0].size
        grid_w = ncol * w + (ncol - 1) * gap
        grid_h = nrow * h + (nrow - 1) * gap
        grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
        for idx, img in enumerate(imgs):
            r, c = divmod(idx, ncol)
            if r < nrow:
                x = c * (w + gap)
                y = r * (h + gap)
                grid.paste(img, (x, y))
        return grid

    gen_grid = make_grid(gen_matrix, nrow, ncol)
    real_grid = make_grid(real_matrix, nrow, ncol)
    # 7. 拼接为左右结构，矩阵间有matrix_gap
    title_font = get_font(font_path, font_size)
    title_font_h = font_size + 6
    total_w = gen_grid.width + real_grid.width + matrix_gap
    total_h = max(gen_grid.height, real_grid.height) + title_font_h + 20
    total = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(total)
    gen_title = "Generated (Predicted Class)"
    real_title = "Real (Ground Truth Class)"
    # 居中写在各自矩阵上方
    draw.text(
        ((gen_grid.width - draw.textlength(gen_title, font=title_font)) // 2, 8),
        gen_title,
        fill=(0, 0, 0),
        font=title_font,
    )
    draw.text(
        (
            gen_grid.width
            + matrix_gap
            + (real_grid.width - draw.textlength(real_title, font=title_font)) // 2,
            8,
        ),
        real_title,
        fill=(0, 0, 0),
        font=title_font,
    )
    # 粘贴图片矩阵
    total.paste(gen_grid, (0, title_font_h + 12))
    total.paste(real_grid, (gen_grid.width + matrix_gap, title_font_h + 12))
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    total.save(save_path)
    print(f"Saved matrix visualization to {save_path}")


if __name__ == "__main__":
    # eval_lfw()
    # eval_face_recognition_lfw()
    # extract_lfw_arcface_features()
    # generate_and_eval_subset_celeba()
    # visualize_gen_vs_real_face_similarity(
    #     gen_dir='./eval_results_celeba_subset_1000_00038_2103',
    #     feature_path='/home/yu/celeba/celeba_arcface_features_align.npz',
    #     identity_file='/home/yu/celeba/identity_CelebA.txt',
    #     arcface_weight='/home/yu/workspace/mia/ckpts/arcface_r100.pth',
    #     out_dir='./gen_vs_real_vis_0038_2103',
    #     max_num=100
    # )
    # compute_lpips_for_gen_vs_real(
    #     gen_dir='./eval_results_celeba_subset_1000_00038_2103',
    #     real_dir='/home/yu/celeba/img_align_celeba_112',
    #     out_csv_name='gen_vs_real_similarity_with_lpips.csv',
    #     lpips_key='lpips',
    #     net='alex'
    # )
    # print_feature_stats_from_npz('/home/yu/workspace/mia/data/lfw_arcface_features.npz')
    # print_feature_stats_from_npz('/home/yu/celeba/celeba_arcface_features_align.npz')
    visualize_gen_vs_real_matrix(
        gen_dir="/home/yu/workspace/mia/outputs/eval_results_celeba_subset_1000_00046_16015",
        csv_path="/home/yu/workspace/mia/outputs/eval_results_celeba_subset_1000_00046_16015/face_recognition_max_sim.csv",
        real_img_root="/home/yu/celeba/img_align_celeba_112",
        identity_file="/home/yu/celeba/identity_CelebA.txt",
        save_path="outputs/visualize_gen_vs_real_00046_16015/gen_vs_real_matrix.png",
        num_images=16,
        nrow=4,
        ncol=4,
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # 你可以换成你喜欢的ttf字体
        font_size=36,  # 字号可调大
    )
