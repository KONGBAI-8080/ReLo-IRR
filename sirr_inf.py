import os
import time
import math
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import contextlib
import pyiqa
import json
from torch.utils.data import Dataset
import sys
from torchvision.transforms import ToTensor  # 导入转换工具
import torchvision.transforms as transforms
import copy
import pandas as pd

# ---------------- LoRA 自定义 forward，与训练脚本一致 ----------------
from train_dreambooth_lora_flux_kontext_sirr import my_lora_fwd

# ---------------- S3DiffImagS3DiffImageEnergyNeweEnergy 定义 ----------------

from train_dreambooth_lora_flux_kontext_sirr import S3DiffImageEnergyNew
# ---------------- RefDet 导入 ----------------

from RDNetRRNetModels.network_RefDet import RefDet

# 1. 配置参数：分桶列表
ASPECT_RATIO_BUCKETS = [
    (672,1568), (688,1504), (720,1456), (752,1392), (800,1328), (832,1248),
    (880,1184), (944,1104), (1024,1024), (1104,944), (1184,880), (1248,832),
    (1328,800), (1392,752), (1456,720), (1504,688), (1568,672)
]
ASPECT_RATIO_BUCKETS_Adaptor = [
    (672,1568),(800,1328), (1568,672) ,(1328,800)
]


# ---------------- 实用函数 ----------------
def find_best_bucket(h, w, buckets=ASPECT_RATIO_BUCKETS):
    src_ratio = h / w
    src_pixels = h * w
    best = None
    best_score = 1e9
    for bw, bh in buckets:
        r = bh / bw
        aspect_diff = abs(math.log(src_ratio / r))
        pixel_diff = abs(math.log(src_pixels / (bh * bw)))
        score = aspect_diff * 100 + pixel_diff
        if score < best_score:
            best_score = score
            best = (bh, bw)
    return best

def resize_to_bucket(img, target_size=None):
    if target_size:
        bw, bh = target_size
    else:
        w, h = img.size
        bh, bw = find_best_bucket(h, w)
    # Resize 到目标桶尺寸
    image = img.resize((bw, bh), Image.BILINEAR)
    # center crop（安全冗余）
    image = transforms.CenterCrop((bh, bw))(image) #(H,W)
    # ToTensor + Normalize
    # tensor = transforms.ToTensor()(image)
    # tensor = transforms.Normalize([0.5], [0.5])(tensor)
    return image, (bw, bh)

def auto_dtype(name: str):
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    mapping = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
               "fp16": torch.float16, "half": torch.float16,
               "fp32": torch.float32, "float32": torch.float32}
    return mapping.get(name, torch.float32)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def collect_lora_layer_names(transformer):
    names = []
    for n, _m in transformer.named_modules():
        if n.endswith(".base_layer"):
            names.append(n[:-11])  # 去掉 .base_layer
    return names

def patch_lora_forward(transformer, lora_layer_names):
    for name, module in transformer.named_modules():
        if name in lora_layer_names:
            if not hasattr(module, "de_mod"):
                module.de_mod = None
            module.forward = my_lora_fwd.__get__(module, module.__class__)

def load_s3_model(path, rank, device, dtype):
    if not path or not os.path.exists(path):
        print(f"[WARN] 未找到 S3DiffImageEnergy_model: {path}")
        return None
    from safetensors.torch import load_file
    state = load_file(path)

    # ----------- 新增：自动推断参数 -----------
    # 默认参数（需与训练时保持一致！）
    default_conv_features = [64, 128, 256, 256]
    # default_global_feature_dim = 128
    default_fusion_hidden_dim = 256*2
    default_num_blocks = 57

    # 尝试从权重推断 rank
    inferred_rank = rank
    for k, v in state.items():
        if v.ndim >= 2 and v.shape[-1] == rank * rank:
            inferred_rank = int(math.isqrt(v.shape[-1]))
            break

    # 尝试推断 num_blocks
    num_blocks = default_num_blocks
    for k, v in state.items():
        if "feature_fusion_img.3.weight" in k and v.ndim == 2:
            # v.shape = (num_blocks, fusion_hidden_dim)
            num_blocks = v.shape[0]
            break

    # 你可以根据实际权重进一步推断 conv_features/global_feature_dim/fusion_hidden_dim
    # 这里只用默认值，确保和训练一致即可

    # ----------- 实例化模型 -----------
    model = S3DiffImageEnergyNew(
        num_blocks=num_blocks,
        lora_rank_transformer=inferred_rank,
        dtype=dtype,
        conv_features=default_conv_features,
        # global_feature_dim=default_global_feature_dim,
        fusion_hidden_dim=default_fusion_hidden_dim,
    )

    # ----------- 加载权重 -----------
    missing = model.load_state_dict(state, strict=False)
    if getattr(missing, "missing_keys", None) and len(missing.missing_keys) > 0:
        print(f"[INFO] S3 模型存在丢失键: {missing.missing_keys}")
    if getattr(missing, "unexpected_keys", None) and len(missing.unexpected_keys) > 0:
        print(f"[INFO] S3 模型存在多余键: {missing.unexpected_keys}")
    model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[INFO] 已加载 S3DiffImageEnergy_model (rank={inferred_rank}, num_blocks={num_blocks})")
    return model, inferred_rank

def load_refdet(args, device, dtype):
    if RefDet is None:
        print("[WARN] 未导入 RefDet，跳过动态调制")
        return None
    if not (args.pretrained_ref_det_path and os.path.exists(args.pretrained_ref_det_path)):
        print("[WARN] 未提供 RefDet 训练权重，跳过")
        return None
    net = RefDet(backbone=args.pretrained_ref_det_backbone_name,
                 proj_planes=args.proj_planes,
                 pred_planes=args.pred_planes,
                 use_pretrained=True,
                 weights_path=args.pretrained_ref_det_backbone_path)
    net.load_state_dict(torch.load(args.pretrained_ref_det_path, map_location="cpu"), strict=True)
    net.to(device=device, dtype=dtype)
    net.eval()
    print("[INFO] 已加载 RefDet")
    return net

def parse_args():
    ap = argparse.ArgumentParser("Kontext LoRA 单图 / 多图反射去除推理 (含 S3 动态调制)")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", default=None, help="包含 pytorch_lora_weights.safetensors 的目录")
    ap.add_argument("--lora_weight", default=None, help="显式指定 LoRA 权重文件名")
    ap.add_argument("--s3_model", default=None, help="S3DiffImageEnergy_model.safetensors 路径")
    ap.add_argument("--input", required=True, help="输入图片文件或目录")
    ap.add_argument("--glob", default="*.jpg,*.png,*.jpeg")
    ap.add_argument("--out_dir", default="inference_outputs")
    ap.add_argument("--prompt", default="reflection removal")
    ap.add_argument("--negative_prompt", default=None)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_image_seed", action="store_true")
    ap.add_argument("--rank", type=int, default=16, help="训练时使用的 LoRA rank（用于 reshape de_mod）")
    ap.add_argument("--dtype", default="auto", choices=["auto","bf16","fp16","fp32"])
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--num_images_per_prompt", type=int, default=1)
    ap.add_argument("--disable_dynamic_mod", action="store_true", help="关闭 S3 + RefDet 动态调制")
    # RefDet & 模型结构参数
    ap.add_argument("--pretrained_ref_det_path", default=None)
    ap.add_argument("--pretrained_ref_det_backbone_path", default=None)
    ap.add_argument("--pretrained_ref_det_backbone_name", default="efficientnet-b3")
    ap.add_argument("--proj_planes", type=int, default=16)
    ap.add_argument("--pred_planes", type=int, default=32)
    ap.add_argument("--max_images", type=int, default=None)
    return ap.parse_args()


class ReflectionRemovalDataset(Dataset):
    """
    单图反射去除数据集类，加载输入图（blended）和对应的标签图（如transmission）
    """
    def __init__(self, 
                 root_in: str,        # 输入图（含反射）文件夹路径
                 glob="*.jpg,*.png,*.jpeg",
                 max_images=500
                 ):
        # 1. 校验文件夹是否存在
        self._check_dir_exists(root_in, "输入图")
        
        # 2. 加载并过滤有效图像文件（仅保留常见图像格式）
        self.img_extensions = tuple(ext.lstrip('*') for ext in glob.split(','))  # 支持的图像格式
        in_files = self._get_valid_image_files(root_in)  # 过滤后的输入文件列表
        self.imgs_in = in_files


        # 4. 样本截断（同步截断输入和标签，避免不匹配）
        if max_images is not None and max_images > 0:
            self.imgs_in = self.imgs_in[:max_images]
                
        # 4. 校验最终样本数是否合法
        if len(self.imgs_in) == 0:
            raise ValueError("数据集加载失败：未找到匹配的输入-标签对，请检查文件命名和路径")

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.imgs_in)

    def __getitem__(self, index):
        """加载第idx个样本（输入图+标签图）"""
        in_img_path = self.imgs_in[index]
        img_name =in_img_path.split('/')[-1]
        return {
            "in_img_path":in_img_path,
            "img_name": img_name,
        }
    
        # -------------------------- 内部工具函数（私有，避免外部调用） --------------------------
    def _check_dir_exists(self, dir_path: str, dir_name: str):
        """校验文件夹是否存在，不存在则报错"""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"{dir_name}文件夹不存在：{dir_path}")

    def _get_valid_image_files(self, dir_path: str) -> list[str]:
        """获取文件夹中所有有效图像文件的路径，按文件名排序"""
        files = []
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            # 过滤：仅保留文件 + 后缀在支持的图像格式中
            if os.path.isfile(fpath) and fname.lower().endswith(self.img_extensions):
                files.append(fpath)
        # 按文件名排序（确保一致性，但后续会通过文件名匹配校验）
        files.sort(key=lambda x: os.path.basename(x))
        if len(files) == 0:
            warnings.warn(f"{dir_path} 中未找到有效图像文件（支持格式：{self.img_extensions}）")
        return files

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = auto_dtype(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    force_hw = None
    if args.width and args.height:
        force_hw = (args.width, args.height)

    # 1) 加载基础 pipeline
    pipe = FluxKontextPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype if dtype != torch.float32 else torch.float32
    )
    pipe.to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    # 2) 加载 LoRA
    if args.lora_dir:
        weight_name = args.lora_weight
        if weight_name is None:
            default_path = Path(args.lora_dir) / "pytorch_lora_weights.safetensors"
            if default_path.exists():
                weight_name = default_path.name
            else:
                cands = list(Path(args.lora_dir).glob("*.safetensors"))
                if cands:
                    weight_name = cands[0].name
                    
        try:
            pipe.load_lora_weights(args.lora_dir, weight_name=weight_name, adapter_name="lora")
            pipe.set_adapters(["lora"], adapter_weights=[1.0])
            print(f"[INFO] 已加载 LoRA: {args.lora_dir}/{weight_name}")
        except Exception as e:
            print(f"[ERROR] 加载 LoRA 失败: {e}")

    # 3) 找出 LoRA 需要动态调制的层并打补丁
    transformer = pipe.transformer
    lora_layer_names = collect_lora_layer_names(transformer)
    patch_lora_forward(transformer, lora_layer_names)
    print(f"[INFO] LoRA 模块数量(需动态调制): {len(lora_layer_names)}")

    # 4) 加载 S3DiffImageEnergy_model
    s3_model = None
    inferred_rank = args.rank
    if not args.disable_dynamic_mod and args.s3_model:
        s3_tuple = load_s3_model(args.s3_model, args.rank, device, dtype)
        if s3_tuple:
            s3_model, inferred_rank = s3_tuple
    if args.disable_dynamic_mod:
        print("[INFO] 已禁用动态调制 (--disable_dynamic_mod)")

    # 5) 加载 RefDet
    refdet = None
    if (not args.disable_dynamic_mod) and s3_model is not None:
        refdet = load_refdet(args, device, dtype)

    # 初始化模块
    vae=pipe.vae
    text_encoder=pipe.text_encoder
    text_encoder_2=pipe.text_encoder_2
    tokenizer=pipe.tokenizer
    tokenizer_2=pipe.tokenizer_2
    transformer=pipe.transformer
    noise_scheduler=pipe.scheduler
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if refdet is not None:
        refdet = refdet.to(device=device,dtype=dtype)
        
    if s3_model is not None:
        s3_model = s3_model.to(device=device,dtype=dtype)
        
    transformer = transformer.to(device=device,dtype=dtype)
    vae = vae.to(device=device,dtype=dtype)

    from kontext_pipeline_sirr import FluxKontextPipeline_New

    new_pipeline = FluxKontextPipeline_New(
        noise_scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        s3_model,
        refdet,
        device,
        dtype
        )
    
    # 6) 收集输入
    eval_loader = ReflectionRemovalDataset(root_in=args.input, glob=args.glob, max_images=args.max_images)
    base_gen = torch.Generator(device=device)

    idx = 1
    for batch in eval_loader:
        try:
            in_img_path = batch["in_img_path"]
            image_name = batch["img_name"]
            
            cur_seed = args.seed + idx if args.per_image_seed else args.seed
            gen = base_gen.manual_seed(cur_seed)
            blended_pil = load_image(str(in_img_path))
            if blended_pil.mode != "RGB":
                blended_pil = blended_pil.convert("RGB")

            bucketed_inputs, (bucket_w, bucket_h) = resize_to_bucket(blended_pil)

            max_area = 1024 ** 2
            if (bucket_w,bucket_h) in ASPECT_RATIO_BUCKETS_Adaptor:
                max_area = 1024 ** 2 + 1024 * 16


            # ----- 推理 -----
            kwargs = {
                "prompt": args.prompt,
                "image": bucketed_inputs,
                "height": bucket_h,
                "width": bucket_w,
                "guidance_scale": args.guidance,
                "num_inference_steps": args.steps,
                "num_images_per_prompt": args.num_images_per_prompt,
                "generator": gen
            }
            if args.negative_prompt is not None:
                kwargs["negative_prompt"] = args.negative_prompt
            with torch.inference_mode():
                result = new_pipeline(**kwargs)
                
            # 修复：展平可能的嵌套列表
            images_list = result.images
            if isinstance(images_list, list) and len(images_list) > 0 and isinstance(images_list[0], list):
                images_list = [img for sublist in images_list for img in sublist]
            
            idx += 1
            os.makedirs(args.out_dir, exist_ok=True)
            images_list[0].save(os.path.join(args.out_dir, f"{image_name[:-4]}.png"))
            print(f"[OK] {image_name} done! Process: {idx}/{len(eval_loader)}")
            
        except Exception as e:
            print(f"[FAIL] {args.input}: {e}")

    print("[DONE] 全部完成.")

if __name__ == "__main__":
    main()


# 推理示例：
# python sirr_inf.py \
#     --base_model /path/model/FLUX.1-Kontext-dev \
#     --lora_dir /path/SIRRCheckPoints/checkpoint-5200 \
#     --s3_model /path/SIRRCheckPoints/checkpoint-5200/S3DiffImageEnergy_model.safetensors \
#     --pretrained_ref_det_path /path/model/RDNetRRNetModels/weights/RD.pth \
#     --pretrained_ref_det_backbone_path /path/model/EfficientNet/efficientnet-b3-5fb5a3c3.pth \
#     --input /path/input_data \
#     --out_dir /path/output \
#     --prompt "reflection removal" \
#     --steps 24 \
#     --rank 16 \
#     --seed 0 \
#     --dtype bf16

