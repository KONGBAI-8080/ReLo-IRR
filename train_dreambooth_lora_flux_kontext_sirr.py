#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import copy
import itertools
import logging
import math
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Tuple, Union
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.models.embeddings import Timesteps,TimestepEmbedding
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, is_wandb_available, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

from safetensors.torch import load_file
from safetensors.torch import save_file


from torch import nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from RDNetRRNetModels.network_RefDet import RefDet

def dict_to_image(img_dict):
    if isinstance(img_dict, dict) and "bytes" in img_dict:
        return Image.open(io.BytesIO(img_dict["bytes"]))
    return img_dict

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux Kontext DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).

Was LoRA for the text encoder enabled? {train_text_encoder}.

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import FluxKontextPipeline
import torch
pipeline = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-kontextflux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)
    pipeline_args_cp = pipeline_args.copy()

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
    with torch.no_grad():
        prompt = pipeline_args_cp.pop("prompt")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt, prompt_2=None)
    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                **pipeline_args_cp,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ref_det_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from ref_det_backbone_model.",
    )
    parser.add_argument(
        "--proj_planes",
        type=int,
        default=16,
        help="a init parameter of ref_det_model.",
    )
    parser.add_argument(
        "--pred_planes",
        type=int,
        default=32,
        help="a init parameter of ref_det_model.",
    )
    parser.add_argument(
        "--pretrained_ref_det_backbone_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from ref_det_backbone_model.",
    )
    parser.add_argument(
        "--pretrained_ref_det_backbone_name",
        type=str,
        default="efficientnet-b3",
        help="Path to pretrained model or model identifier from ref_det_model_name.",
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_encode_mode",
        type=str,
        default="mode",
        choices=["sample", "mode"],
        help="VAE encoding mode.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="Validation image to use (during I2I fine-tuning) to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-kontext-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_rotate",
        action="store_true",
        help="whether to randomly rotate images",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=24,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--s3diff_lr", type=float, default=None, help="S3DiffImageEnergy_model 学习率（默认=主 learning_rate）")
    parser.add_argument("--freeze_s3diff", action="store_true", help="冻结 S3DiffImageEnergy_model（保持旧行为，不训练）")
    parser.add_argument("--grad_debug_steps", type=int, default=0, help="前 N 个优化 step 打印梯度统计 (0 关闭)")
    parser.add_argument("--anomaly_detect", action="store_true", help="开启 autograd 异常检测(低速)")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
        if args.cond_image_column is not None:
            raise ValueError("Prior preservation isn't supported with I2I training.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.cond_image_column is not None:
        assert args.image_column is not None
        assert args.caption_column is not None
        assert args.dataset_name is not None
        assert not args.train_text_encoder
        if args.validation_prompt is not None:
            assert args.validation_image is None and os.path.exists(args.validation_image)

    return args


def _iter_lora_modules(transformer, transformer_lora_layers):
    for name, module in transformer.named_modules():
        if name in transformer_lora_layers:
            yield module

def find_nearest_bucket(height, width, buckets):
    aspect = height / width
    # Find bucket with closest aspect ratio
    closest_bucket = min(buckets, key=lambda bucket: abs(bucket[0]/bucket[1] - aspect))
    
    # If multiple buckets have same aspect difference, choose smallest area that fits
    area = height * width
    fitting_buckets = [b for b in buckets if b[0] >= height and b[1] >= width]
    if fitting_buckets:
        return min(fitting_buckets, key=lambda b: b[0]*b[1])
    
    # Otherwise use closest aspect
    return closest_bucket


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        repeats=1,
        center_crop=False,
        random_flip=False,
        random_rotate=False,
        buckets=None,
        args=None,
    ):
        self.center_crop = center_crop
        self.random_flip = args.random_flip if args is not None else random_flip
        self.random_rotate = args.random_rotate if args is not None else random_rotate
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.buckets = buckets
        self.args = args
        
        # Create cache directory if using caching
        cache_dir = getattr(args, "cache_dir", None)
        if not cache_dir:
            cache_dir = "./dataset_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.size_cache_path = self.cache_dir / "image_sizes.npy"

        # Process dataset based on source
        if args.dataset_name is not None:
            self._init_hf_dataset(repeats)
        else:
            self._init_local_dataset(instance_data_root, repeats)

        # Precompute bucket indices in parallel
        self._precompute_bucket_indices()

        # Initialize class images
        self._init_class_images(class_data_root, class_num)

    def _init_hf_dataset(self, repeats):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for HF datasets")

        dataset = load_dataset(
            "imagefolder",
            data_dir=args.dataset_name,
        )
        self.dataset = dataset["train"]
        self.num_samples = len(self.dataset)
        
        column_names = self.dataset.column_names
        self.image_column = self.args.image_column or column_names[0]
        self.cond_image_column = self.args.cond_image_column
        
        # Handle custom prompts
        if self.args.caption_column:
            captions = self.dataset[self.args.caption_column]
            self.custom_instance_prompts = list(itertools.chain.from_iterable(
                itertools.repeat(caption, repeats) for caption in captions
            ))
            
        # Create repeated indices
        self.instance_indices = list(itertools.chain.from_iterable(
            itertools.repeat(i, repeats) for i in range(self.num_samples)
        ))

    def _init_local_dataset(self, instance_data_root, repeats):
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")
            
        # List all image paths
        image_paths = [p for p in self.instance_data_root.iterdir() if p.is_file()]
        self.image_paths = list(itertools.chain.from_iterable(
            itertools.repeat(path, repeats) for path in image_paths
        ))
        self.num_samples = len(self.image_paths)

    def _precompute_bucket_indices(self):
        """Precompute bucket indices in parallel with progress tracking"""
        self.bucket_indices = [None] * self.num_samples
        
        # Try to load cached sizes
        if self.size_cache_path.exists():
            try:
                sizes = np.load(self.size_cache_path, allow_pickle=True)
                self._process_sizes(sizes)
                logger.info(f"Loaded cached image sizes from {self.size_cache_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load size cache: {e}")
        
        # Compute sizes in parallel
        sizes = [None] * self.num_samples
        with ThreadPoolExecutor(max_workers=48) as executor:
            futures = {}
            for i in range(self.num_samples):
                futures[executor.submit(self._get_image_size, i)] = i
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc="Computing image sizes"):
                idx = futures[future]
                sizes[idx] = future.result()
        
        # Save to cache
        np.save(self.size_cache_path, sizes, allow_pickle=True)
        self._process_sizes(sizes)
        # logger.warning(f"target_sizes: {self.target_sizes}")

    def _process_sizes(self, sizes):
        """Process precomputed sizes into bucket indices"""
        self.bucket_indices = [0] * self.num_samples
        self.target_sizes = [None] * self.num_samples
        # logging.warning(f"num_samples: {self.num_samples}")
        # logging.warning(f"target_size: {self.target_sizes}")
        
        # Compute bucket indices in parallel
        with ThreadPoolExecutor(max_workers=48) as executor:
            futures = {}
            for i, size in enumerate(sizes):
                if size is None:
                    continue
                futures[executor.submit(
                    find_nearest_bucket, size[0], size[1], self.buckets
                )] = i
            
            for future in tqdm(as_completed(futures), total=len(futures),
                            desc="Computing bucket assignments"):
                idx = futures[future]
                bucket = future.result()  # 这里bucket是tuple
                # logging.warning(f"bucket: {bucket}")
                if bucket in self.buckets:
                    bucket_idx = self.buckets.index(bucket)
                else:
                    bucket_idx = 0  # fallback
                self.bucket_indices[idx] = bucket_idx
                self.target_sizes[idx] = bucket

    def _get_image_size(self, index):
        """Get image size without loading full image"""
        if hasattr(self, 'dataset'):
            return self._get_hf_image_size(index)
        return self._get_local_image_size(index)

    def _get_hf_image_size(self, index):
        """Get image size from HF dataset"""
        try:
            img_data = self.dataset[index][self.image_column]
            if isinstance(img_data, dict):  # Bytes format
                with Image.open(io.BytesIO(img_data['bytes'])) as img:
                    return img.size
            else:  # PIL Image
                return img_data.size
        except Exception as e:
            logger.error(f"Error getting size for index {index}: {e}")
            return (512, 512)  # Fallback size

    def _get_local_image_size(self, index):
        """Get image size from local file"""
        try:
            with Image.open(self.image_paths[index]) as img:
                img = exif_transpose(img)
                return img.size
        except Exception as e:
            logger.error(f"Error getting size for {self.image_paths[index]}: {e}")
            return (512, 512)  # Fallback size

    def _load_image(self, index):
        """Load image for given index"""
        if hasattr(self, 'dataset'):
            return self._load_hf_image(index)
        return self._load_local_image(index)

    def _load_hf_image(self, index):
        orig_idx = self.instance_indices[index]
        img = self.dataset[orig_idx][self.image_column]
        
        if isinstance(img, dict):
            img = Image.open(io.BytesIO(img['bytes']))
        img = exif_transpose(img)
        
        if not img.mode == "RGB":
            img = img.convert("RGB")
            
        cond_img = None
        if self.cond_image_column:
            cond_img = self.dataset[orig_idx][self.cond_image_column]
            if isinstance(cond_img, dict):
                cond_img = Image.open(io.BytesIO(cond_img['bytes']))
            cond_img = exif_transpose(cond_img)
            if cond_img and not cond_img.mode == "RGB":
                cond_img = cond_img.convert("RGB")
                
        return img, cond_img

    def _load_local_image(self, index):
        img = Image.open(self.image_paths[index])
        img = exif_transpose(img)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        return img, None

    def _init_class_images(self, class_data_root, class_num):
        """Initialize class images with caching"""
        if class_data_root is None:
            self.class_data_root = None
            self.num_class_images = 0
            self.class_image_paths = []
            return

        self.class_data_root = Path(class_data_root)
        self.class_data_root.mkdir(parents=True, exist_ok=True)
        self.class_image_paths = sorted(list(self.class_data_root.iterdir()))
        
        # Create bucket cache for class images
        class_cache_path = self.cache_dir / "class_buckets.npy"
        if class_cache_path.exists():
            try:
                self.class_buckets = np.load(class_cache_path, allow_pickle=True)
                logger.info(f"Loaded class bucket cache from {class_cache_path}")
            except:
                self.class_buckets = self._precompute_class_buckets()
                np.save(class_cache_path, self.class_buckets)
        else:
            self.class_buckets = self._precompute_class_buckets()
            np.save(class_cache_path, self.class_buckets)

        self.num_class_images = min(len(self.class_image_paths), class_num or float('inf'))
        if self.num_class_images < len(self.class_image_paths):
            self.class_image_paths = random.sample(self.class_image_paths, self.num_class_images)

    def _precompute_class_buckets(self):
        """Precompute class image buckets in parallel"""
        num_class = len(self.class_image_paths)
        sizes = [None] * num_class
        with ThreadPoolExecutor(max_workers=48) as executor:
            futures = {executor.submit(self._get_class_size, i): i for i in range(num_class)}
            for future in as_completed(futures):
                i = futures[future]
                sizes[i] = future.result()

        buckets = [None] * num_class
        with ThreadPoolExecutor(max_workers=48) as executor:
            futures = {}
            for i, size in enumerate(sizes):
                if size is None:
                    continue
                futures[executor.submit(
                    find_nearest_bucket, size[0], size[1], self.buckets
                )] = i
            
            for future in as_completed(futures):
                i = futures[future]
                buckets[i] = self.buckets[future.result()]
                
        return buckets

    def _get_class_size(self, index):
        try:
            with Image.open(self.class_image_paths[index]) as img:
                img = exif_transpose(img)
                return img.size
        except Exception as e:
            logger.error(f"Error getting class image size: {e}")
            return (512, 512)

    def __len__(self):
        return max(self.num_class_images, self.num_samples) if self.class_data_root else self.num_samples

    def __getitem__(self, index):
        instance_idx = index % self.num_samples
        image, cond_image = self._load_image(instance_idx)
        
        # logger.warning(f"instance_idx: {str(instance_idx)}")
        # logger.warning(f"target_sizes: {self.target_sizes}")
        # Get target size for this index
        target_size = self.target_sizes[instance_idx]
        # logger.warning(f"target_size: {target_size}")
        if target_size is None:
            logger.warning(f"Using fallback size for index {index}")
            target_size = (512, 512)
        
        # Apply transformations
        image, cond_image = self.paired_transform(
            image, 
            dest_image=cond_image,
            size=target_size,
            center_crop=self.center_crop,
            random_flip=self.random_flip,
            random_rotate=self.random_rotate
        )
        
        # Create sample dictionary
        sample = {
            "instance_images": image,
            "bucket_idx": self.bucket_indices[instance_idx],
        }
        
        if cond_image is not None:
            sample["cond_images"] = cond_image
            
        # Add prompt
        if self.custom_instance_prompts:
            sample["instance_prompt"] = self.custom_instance_prompts[instance_idx]
        else:
            sample["instance_prompt"] = self.instance_prompt
            
        # Add class images if available
        if self.class_data_root and index < self.num_class_images:
            class_img = Image.open(self.class_image_paths[index])
            class_img = exif_transpose(class_img)
            if not class_img.mode == "RGB":
                class_img = class_img.convert("RGB")
                
            # Transform to target size
            h, w = self.class_buckets[index]
            class_img = transforms.Compose([
                transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((h, w)) if self.center_crop else transforms.RandomCrop((h, w)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])(class_img)
            
            sample["class_images"] = class_img
            sample["class_prompt"] = self.class_prompt

        return sample

    def paired_transform(self, image, dest_image=None, size=(512, 512), 
                        center_crop=False, random_flip=False, random_rotate=False):
        # Convert size to (height, width)
        height, width = size
        
        # Resize with same interpolation
        resize = transforms.Resize((height, width), 
                                  interpolation=transforms.InterpolationMode.BILINEAR)
        image = resize(image)
        if dest_image is not None:
            dest_image = resize(dest_image)
            
        # Crop - same for both images
        if center_crop:
            crop = transforms.CenterCrop((height, width))
            image = crop(image)
            if dest_image is not None:
                dest_image = crop(dest_image)
        else:
            # Same random crop for both images
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(height, width))
            image = TF.crop(image, i, j, h, w)
            if dest_image is not None:
                dest_image = TF.crop(dest_image, i, j, h, w)
                
        # Random flip - same for both
        if random_flip and random.random() < 0.5:
            image = TF.hflip(image)
            if dest_image is not None:
                dest_image = TF.hflip(dest_image)
        
        # Random rotate - same for both
        # 只能旋转180度，因为图像的size不是正方形，否则会有黑边
        if random_rotate and random.random() < 0.5:
            image = TF.rotate(image, 180)
            if dest_image is not None:
                dest_image = TF.rotate(dest_image, 180)
                
        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5], [0.5])(image)
        
        if dest_image is not None:
            dest_image = transforms.ToTensor()(dest_image)
            dest_image = transforms.Normalize([0.5], [0.5])(dest_image)
            return image, dest_image
            
        return image, None

def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    
    if any("cond_images" in example for example in examples):
        cond_pixel_values = [example["cond_images"] for example in examples]
        cond_pixel_values = torch.stack(cond_pixel_values)
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
        batch["cond_pixel_values"] = cond_pixel_values
        
    return batch


class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset: DreamBoothDataset, batch_size: int, drop_last: bool = False):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by bucket
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets))]
        for idx, bucket_idx in enumerate(self.dataset.bucket_indices):
            self.bucket_indices[bucket_idx].append(idx)

        self.sampler_len = 0
        self.batches = []

        # Pre-generate batches for each bucket
        for indices_in_bucket in self.bucket_indices:
            # Shuffle indices within the bucket
            random.shuffle(indices_in_bucket)
            # Create batches
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip partial batch if drop_last is True
                self.batches.append(batch)
                self.sampler_len += 1  # Count the number of batches

    def __iter__(self):
        # Shuffle the order of the batches each epoch
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.sampler_len


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example





@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, dtype=torch.bfloat16, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                m.weight.data = m.weight.data.to(dtype=dtype)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
                    m.bias.data = m.bias.data.to(dtype=dtype)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                m.weight.data = m.weight.data.to(dtype=dtype)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
                    m.bias.data = m.bias.data.to(dtype=dtype)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                m.weight.data = m.weight.data.to(dtype=dtype)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
                    m.bias.data = m.bias.data.to(dtype=dtype)


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        # logging.warning(f"*****************:t5 text_inputs: {text_inputs}")

        text_input_ids = text_inputs.input_ids
        # logging.warning(f"*****************:t5 text_input_ids: {text_input_ids}")

    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    # logging.warning(f"*****************:t5 prompt_embeds: {prompt_embeds.size()}")

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    # logging.warning(f"*****************:t5 prompt_embeds: {prompt_embeds.size()}")

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        # logging.warning(f"*****************:clip text_inputs: {text_inputs}")
        
        text_input_ids = text_inputs.input_ids
        # logging.warning(f"*****************:clip text_input_ids: {text_input_ids}")

    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    # logging.warning(f"*****************:clip prompt_embeds.last_hidden_state: {prompt_embeds.last_hidden_state.size()}")

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    # logging.warning(f"*****************:clip prompt_embeds: {prompt_embeds}")
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    # logging.warning(f"*****************:clip prompt_embeds: {prompt_embeds.size()}")

    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    # logging.warning(f"*****************:clip prompt_embeds: {prompt_embeds.size()}")

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids



def my_lora_fwd(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:


    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # 将输入转换为 lora_A 的计算 dtype，减少 dtype 转换
            x_lora = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                _tmp = lora_A(dropout(x_lora))
                de_mod = getattr(self, "de_mod", None)
                # print(f"*****************:de_mod: {de_mod}")
                if de_mod is None:
                    # print("LoRA forward: de_mod is None, using standard LoRA path.")
                    result = result + lora_B(_tmp) * scaling
                else:
                    try:
                        if de_mod.device != _tmp.device or de_mod.dtype != _tmp.dtype:
                            de_mod_local = de_mod.to(device=_tmp.device, dtype=_tmp.dtype)
                        else:
                            de_mod_local = de_mod
                        if isinstance(lora_A, torch.nn.Conv2d):
                            # print("LoRA forward: using einsum for Conv2d.")
                            _tmp2 = torch.einsum("...khw,...kr->...rhw", _tmp, de_mod_local)
                        elif isinstance(lora_A, torch.nn.Linear):
                            # print("LoRA forward: using einsum for Linear.")
                            _tmp2 = torch.einsum("...lk,...kr->...lr", _tmp, de_mod_local)
                        else:
                            # print("LoRA forward: unsupported lora_A type, fallback to standard.")
                            result = result + lora_B(_tmp) * scaling
                            continue
                        result = result + lora_B(_tmp2) * scaling
                    except Exception as e:
                        # print(f"LoRA forward: exception occurred: {e}, fallback to standard.")
                        result = result + lora_B(_tmp) * scaling

    return result

##############################修改，AdaLayerNormZeroSingle， 添加了时间依赖 #################################
from typing import Any, List, Optional, Tuple, Union
class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa
        
class S3DiffImageEnergyNew(torch.nn.Module):
    def __init__(self,
                 # 共用参数
                 num_blocks=57,
                 lora_rank_transformer=16,   # LoRA秩（两个分支输出维度依赖）
                 dtype=torch.bfloat16,
                 
                 # S3DiffImage 分支参数
                 input_channels=1,           # 输入图像通道数
                 conv_features=[64, 128, 256, 256],  # 卷积层特征通道数
                #  global_feature_dim=128,     # 全局特征维度
                 fusion_hidden_dim=256*2,      # 图像分支融合层隐藏维度
                 sigmas_feature_dim=128,     # 时间步长特征维度
                 ):
        super().__init__()
        
        # -------------------------- 2. 图像分支（原 S3DiffImage） --------------------------
        self.output_dim_img = lora_rank_transformer ** 2  # 256
        self.num_blocks = num_blocks
        # 2.1 卷积特征提取器
        self.conv_feature_extractor = nn.Sequential(
            # 第1组
            nn.Conv2d(input_channels, conv_features[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_features[0]),  # 批量归一化，加速训练
            nn.ReLU(True),
            # 第1次下采样
            nn.Conv2d(conv_features[0], conv_features[1], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_features[1]),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第2次下采样
            nn.Conv2d(conv_features[1], conv_features[2], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_features[2]),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第3次下采样
            nn.Conv2d(conv_features[2], conv_features[3], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_features[3]),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.time_proj = Timesteps(num_channels=sigmas_feature_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=sigmas_feature_dim, time_embed_dim=self.output_dim_img)

        # 4. 添加层归一化
        self.ada_norm = AdaLayerNormZeroSingle(self.output_dim_img)

        # 2.2 全局特征处理
        # 改进后：增加批归一化+Dropout，提升泛化性；可选2层线性增强表达
        self.global_feature_processor = nn.Sequential(
            # 第一步：全局池化后先归一化，稳定特征分布（关键！池化后特征方差可能变化）
            nn.BatchNorm1d(conv_features[-1], dtype=dtype),  # 输入是(B, 256)，用1dBatchNorm
            # 第二步：可选Dropout抑制过拟合（尤其若数据量小）
            nn.Dropout(p=0.1, inplace=True),  # p建议0.1~0.3，避免过度丢弃
            # 第三步：2层线性+ReLU，增强特征抽象能力（避免单一线性层的局限性）
            nn.Linear(conv_features[-1], conv_features[-1] * 2, dtype=dtype),  # 256->256*2
            nn.ReLU(True),
            nn.Linear(conv_features[-1] * 2, self.output_dim_img, dtype=dtype),  # 256*2→256（保持目标维度）
            
        )
        
        # 特征融合与输出
        self.feature_fusion_img = nn.Sequential(
            # 2. 特征映射：128→256（匹配 LoRA 输出维度需求）
            nn.Linear(self.output_dim_img, fusion_hidden_dim, dtype=dtype),  # 128→256
            nn.ReLU(True),
            # 3. 输出最终调制嵌入（256→256，与 self.output_dim_img 匹配）
            nn.Linear(fusion_hidden_dim, self.output_dim_img, dtype=dtype),  # 256→256
            # 5. 可选：Tanh 压缩输出范围，避免 LoRA 调制权重过大
            # nn.Tanh()
        )
        
        self.norm = nn.LayerNorm(self.output_dim_img, dtype=dtype)

        # -------------------------- 4. 初始化权重 --------------------------
        # 统一初始化所有可训练模块
        default_init_weights(
            [self.conv_feature_extractor, self.global_feature_processor, self.feature_fusion_img],  # 共用嵌入层也需初始化
            1e-5, dtype=dtype
        )

    # -------------------------- 图像分支前向传播（原 S3DiffImage.forward） --------------------------
    def forward(self, image, timestep):
        # 输入: image - (B, 1, H, W) 单通道黑白图像
        # 1. 卷积特征提取
        conv_features = self.conv_feature_extractor(image)  # (B, 256, H', W')
        
        # # 2. 特征侧零卷积（无残差，直接用调制后特征） 先不加 缩放
        # modulated_features = self.zero_conv_feature(conv_features)  # 初始为0

        # 2. 全局特征（空间平均）
        global_features = torch.mean(conv_features, dim=[2, 3])  # (B, 256)
        global_features = self.global_feature_processor(global_features)  # (B, 128)

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=image.dtype))

        global_features_ada, gate_scale = self.ada_norm(global_features, emb=timesteps_emb)

        # 3. 先对全局特征做一次映射，再重复到 num_blocks 个维度
        unified_embedding = self.feature_fusion_img(global_features_ada)
        # 4. ada_norm
        unified_embedding_norm = self.norm(global_features + gate_scale * unified_embedding)
        # 4. 将统一嵌入重复到 num_blocks 个维度，匹配下游输出格式 (B, num_blocks, 256)
        modulation_embedding_img = unified_embedding_norm.unsqueeze(1).repeat(1, self.num_blocks, 1)  # (B, 57, 256)
        return modulation_embedding_img

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=None,
        log_with=args.report_to, # 开启wandb
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16

            transformer = FluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=args.revision,
                variant=args.variant,
            )
            pipeline = FluxKontextPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            free_memory()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    S3DiffImageEnergy_model = S3DiffImageEnergyNew(lora_rank_transformer=args.rank, dtype=weight_dtype)
    
    if args.freeze_s3diff:
        for p in S3DiffImageEnergy_model.parameters():
            p.requires_grad = False
    
    net_Det = RefDet(backbone=args.pretrained_ref_det_backbone_name, 
                     proj_planes=args.proj_planes, 
                     pred_planes=args.pred_planes,
                     use_pretrained=True,
                     weights_path=args.pretrained_ref_det_backbone_path,
                     )
    net_Det.load_state_dict(torch.load(args.pretrained_ref_det_path), strict=True)
    net_Det.eval() # 保证每次输出的图像都是一致的，而不是随机的

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    S3DiffImageEnergy_model.to(accelerator.device, dtype=weight_dtype)
    net_Det.to(accelerator.device, dtype=weight_dtype)


    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "proj_mlp",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    transformer_lora_layers = []
    for name, module in transformer.named_modules():
        if 'base_layer' in name:
            transformer_lora_layers.append(name[:-len(".base_layer")])

    for name, module in transformer.named_modules():
        if name in transformer_lora_layers:
            if not hasattr(module, "de_mod"):
                module.de_mod = None
            module.forward = my_lora_fwd.__get__(module, module.__class__)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            print(len(models))
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(S3DiffImageEnergy_model))):
                    # 保存自定义辅助模型（完整模型，非LoRA）
                    S3DiffImageEnergy_model_unwrapped = unwrap_model(model)
                    # 保存完整模型的状态字典（而非LoRA）
                    S3DiffImageEnergy_model_state_dict = S3DiffImageEnergy_model_unwrapped.state_dict()  # 完整权重
                    S3DiffImageEnergy_path = os.path.join(output_dir, "S3DiffImageEnergy_model.safetensors")
                    save_file(S3DiffImageEnergy_model_state_dict, S3DiffImageEnergy_path)
                    print(f"自定义辅助模型S3DiffImageEnergy_model （完整）已保存至：{S3DiffImageEnergy_path}")
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["text_encoder"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")


            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

            # 从 weights 中移除已处理的模型（避免 Accelerator 重复保存）
            while weights:
                weights.pop()


    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None
        S3DiffImageEnergy_model_ = None
        # print(len(models))
        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(S3DiffImageEnergy_model))):
                S3DiffImageEnergy_model_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the transformer model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # ---------------------- 4. 加载自定义辅助模型（完整模型，非LoRA） ----------------------
        S3DiffImageEnergy_model_path = os.path.join(input_dir, "S3DiffImageEnergy_model.safetensors")
        if os.path.exists(S3DiffImageEnergy_model_path):
            # 直接加载完整状态字典到模型
            S3DiffImageEnergy_model_state_dict = load_file(S3DiffImageEnergy_model_path)
            S3DiffImageEnergy_model_.load_state_dict(S3DiffImageEnergy_model_state_dict)  # 完整模型加载
            print(f"自定义辅助模型S3DiffImage （完整）已加载：{S3DiffImageEnergy_model_path}")
        else:
            print(f"未找到自定义辅助模型S3DiffImageEnergy_model文件：{S3DiffImageEnergy_model_path}（跳过加载）")



        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


    # for name, module in transformer.named_modules():
    #     print(name)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))

    # 优化参数设置
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    if args.train_text_encoder:
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize.append(text_parameters_one_with_lr)

    # S3DiffImageEnergy_model参数
    s3diff_params = list(filter(lambda p: p.requires_grad, S3DiffImageEnergy_model.parameters()))
    if s3diff_params:
        s3diff_lr = args.s3diff_lr if args.s3diff_lr is not None else args.learning_rate
        params_to_optimize.append({"params": s3diff_params, "lr": s3diff_lr})

    # 参数统计
    def print_trainable_params(params, name):
        num = sum(p.numel() for p in params)
        print(f"{name} 可训练参数数量: {num}")
        return num

    print("##############LoRA参数数量统计##############")
    num_trainable_params = print_trainable_params(transformer_lora_parameters, "FluxTransformer2DModel LoRA")
    if args.train_text_encoder:
        num_trainable_params_text_encoder = print_trainable_params(text_lora_parameters_one, "TextEncoderOne LoRA")
        num_trainable_params += num_trainable_params_text_encoder
    num_s3diff_params = print_trainable_params(s3diff_params, "S3DiffImageEnergy_model")
    num_trainable_params += num_s3diff_params
    print(f"总可训练参数数量: {num_trainable_params}")
    print("#########################################")


    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}")

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        buckets=buckets,
        repeats=args.repeats,
        center_crop=args.center_crop,
        args=args,
    )
    if args.cond_image_column is not None:
        logger.info("I2I fine-tuning enabled.")
    batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True if args.dataloader_num_workers > 0 else False,
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    # 只需处理一次 embedding，如果没有自定义 caption
    if not train_dataset.custom_instance_prompts and not args.train_text_encoder:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )
        if args.with_prior_preservation:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )
            prompt_embeds = torch.cat([instance_prompt_hidden_states, class_prompt_hidden_states], dim=0)
            pooled_prompt_embeds = torch.cat([instance_pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
            text_ids = torch.cat([instance_text_ids, class_text_ids], dim=0)
        else:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids

            
    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        text_encoder_one.cpu(), text_encoder_two.cpu()
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt, max_sequence_length=77)
            tokens_two = tokenize_prompt(
                tokenizer_two, args.instance_prompt, max_sequence_length=args.max_sequence_length
            )
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt, max_sequence_length=77)
                class_tokens_two = tokenize_prompt(
                    tokenizer_two, args.class_prompt, max_sequence_length=args.max_sequence_length
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    elif train_dataset.custom_instance_prompts and not args.train_text_encoder:
        # 直接只计算一次 embedding，所有 batch 共享
        single_prompt = train_dataset.custom_instance_prompts[0]
        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
            [single_prompt], text_encoders, tokenizers
        )
        cached_text_embeddings = [(prompt_embeds, pooled_prompt_embeds, text_ids)] * len(train_dataloader)

        if args.validation_prompt is None:
            text_encoder_one.cpu(), text_encoder_two.cpu()
            del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
            free_memory()

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    has_image_input = args.cond_image_column is not None
    if args.cache_latents:
        latents_cache = []
        cond_latents_cache = []
        cond_reflection_cache = []  # 新增：缓存 net_Det 输出（无梯度需求）
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                if args.vae_encode_mode == "sample":
                    lat = vae.encode(batch["pixel_values"]).latent_dist.sample()
                else:
                    lat = vae.encode(batch["pixel_values"]).latent_dist.mode()
                latents_cache.append(lat.detach().cpu())

                if has_image_input:
                    batch["cond_pixel_values"] = batch["cond_pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    if args.vae_encode_mode == "sample":
                        clat = vae.encode(batch["cond_pixel_values"]).latent_dist.sample()
                    else:
                        clat = vae.encode(batch["cond_pixel_values"]).latent_dist.mode()
                    cond_latents_cache.append(clat.detach().cpu())
                    # 仅缓存 net_Det 输出（不缓存 transformer_embeds，保持 S3Diff 可训练）
                    cond_refl = net_Det(batch["cond_pixel_values"])  # net_Det 已 eval + no_grad
                    cond_reflection_cache.append(cond_refl.detach().cpu())

        if args.validation_prompt is None:
            vae.cpu()
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    S3DiffImageEnergy_model = accelerator.prepare(S3DiffImageEnergy_model)

    if args.anomaly_detect and accelerator.is_main_process:
        torch.autograd.set_detect_anomaly(True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-kontext-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0



    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    has_guidance = unwrap_model(transformer).config.guidance_embeds
    for epoch in range(first_epoch, args.num_train_epochs):
        
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
        
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.append(text_encoder_one)
            # 加入 S3DiffImageEnergy_model，确保与 Transformer 同步梯度累积
            if (not args.freeze_s3diff) and any(p.requires_grad for p in S3DiffImageEnergy_model.parameters()):
                models_to_accumulate.append(S3DiffImageEnergy_model)

            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                # 只在有自定义 caption 时才重新计算 embedding
                if train_dataset.custom_instance_prompts:
                    if not args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = cached_text_embeddings[step]
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                        tokens_two = tokenize_prompt(
                            tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
                        )
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=prompts,
                        )
                else:
                    # 直接复用预先计算好的 embedding
                    pass  # prompt_embeds, pooled_prompt_embeds, text_ids 已在前面准备好


                # Convert images to latent space
                if args.cache_latents:
                    model_input = latents_cache[step].to(accelerator.device, dtype=weight_dtype)
                    if has_image_input:
                        cond_model_input = cond_latents_cache[step].to(accelerator.device, dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    if has_image_input:
                        cond_pixel_values = batch["cond_pixel_values"].to(device=accelerator.device, non_blocking=True, dtype=weight_dtype)
                        
                    if args.vae_encode_mode == "sample":
                        model_input = vae.encode(pixel_values).latent_dist.sample()
                        if has_image_input:
                            cond_model_input = vae.encode(cond_pixel_values).latent_dist.sample()
                    else:
                        model_input = vae.encode(pixel_values).latent_dist.mode()
                        if has_image_input:
                            cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()

                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)
                if has_image_input:
                    cond_model_input = (cond_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                    cond_model_input = cond_model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                if has_image_input:
                    cond_latents_ids = FluxKontextPipeline._prepare_latent_image_ids(
                        cond_model_input.shape[0],
                        cond_model_input.shape[2] // 2,
                        cond_model_input.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    cond_latents_ids[..., 0] = 1
                    latent_image_ids = torch.cat([latent_image_ids, cond_latents_ids], dim=0)


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)

                if args.cache_latents:
                    if has_image_input:
                        # 重新前向 S3DiffImageEnergy_model，保持梯度链路
                        cond_reflection_image = cond_reflection_cache[step].to(accelerator.device, dtype=weight_dtype)
                        transformer_embeds = S3DiffImageEnergy_model(cond_reflection_image, timesteps)
                        model = transformer.module if hasattr(transformer, "module") else transformer
                        for layer_name, module in model.named_modules():
                            if layer_name in transformer_lora_layers:
                                split_name = layer_name.split(".")
                                if split_name[0] == 'transformer_blocks':
                                    block_id = int(split_name[1])
                                elif split_name[0] == 'single_transformer_blocks':
                                    block_id = int(split_name[1]) + 19
                                else:
                                    continue
                                transformer_embed = transformer_embeds[:, block_id]
                                sample_lora_A = next(iter(module.lora_A.values()))
                                module.de_mod = transformer_embed.reshape(-1, args.rank, args.rank).to(
                                    dtype=sample_lora_A.weight.dtype,
                                    device=sample_lora_A.weight.device
                                )
                else:
                    if has_image_input:
                        # 推理部分全部不构建计算图
                        with torch.no_grad():
                            cond_reflection_image = net_Det(cond_pixel_values)

                        transformer_embeds = S3DiffImageEnergy_model(cond_reflection_image, timesteps)
                        transformer_embeds = transformer_embeds.to(accelerator.device, dtype=weight_dtype)
                        model = transformer.module if hasattr(transformer, "module") else transformer
                        for layer_name, module in model.named_modules():
                            if layer_name in transformer_lora_layers:
                                split_name = layer_name.split(".")
                                if split_name[0] == 'transformer_blocks':
                                    block_id = int(split_name[1])
                                elif split_name[0] == 'single_transformer_blocks':
                                    block_id = int(split_name[1]) + 19
                                else:
                                    continue
                                transformer_embed = transformer_embeds[:, block_id]  # (B, rank^2)
                                try:
                                    sample_lora_A = next(iter(module.lora_A.values()))
                                    target_dtype = sample_lora_A.weight.dtype
                                    target_device = sample_lora_A.weight.device
                                except Exception:
                                    target_dtype = transformer_embed.dtype
                                    target_device = transformer_embed.device
                                # 不 detach，不包 no_grad，保持计算图
                                module.de_mod = transformer_embed.reshape(-1, args.rank, args.rank).to(
                                    dtype=target_dtype, device=target_device
                                )

                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                orig_inp_shape = packed_noisy_model_input.shape
                if has_image_input:
                    packed_cond_input = FluxKontextPipeline._pack_latents(
                        cond_model_input,
                        batch_size=cond_model_input.shape[0],
                        num_channels_latents=cond_model_input.shape[1],
                        height=cond_model_input.shape[2],
                        width=cond_model_input.shape[3],
                    )
                    packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_input], dim=1)

                # Kontext always has guidance
                guidance = None
                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])

                # Predict the noise residual
                with accelerator.autocast():
                    model_pred = transformer(
                        hidden_states=packed_noisy_model_input,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]
                if has_image_input:
                    model_pred = model_pred[:, : orig_inp_shape[1]]
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = []
                    params_to_clip.extend(list(transformer.parameters()))
                    if args.train_text_encoder:
                        params_to_clip.extend(list(text_encoder_one.parameters()))
                    if (not args.freeze_s3diff) and s3diff_params:
                        params_to_clip.extend(list(S3DiffImageEnergy_model.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 释放 de_mod（避免持有上一轮图，下一轮重新写入）
                for m in _iter_lora_modules(transformer, transformer_lora_layers):
                    m.de_mod = None

                # 释放大对象，避免显存峰值累积
                try:
                    del model_pred, noisy_model_input, packed_noisy_model_input, packed_cond_input, latent_image_ids, noise, target, sigmas, weighting
                except Exception:
                    pass
                for m in _iter_lora_modules(transformer, transformer_lora_layers): m.de_mod = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            modules_to_save["text_encoder"] = text_encoder_one
        else:
            text_encoder_lora_layers = None

        FluxKontextPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        S3DiffImageEnergy_model = unwrap_model(S3DiffImageEnergy_model)
        if args.upcast_before_saving:
            S3DiffImageEnergy_model.to(torch.float32)  # 保存前升为 FP32（避免精度损失）
        else:
            S3DiffImageEnergy_model = S3DiffImageEnergy_model.to(weight_dtype)
        # 保存完整模型的状态字典（而非LoRA）
        S3DiffImageEnergy_model_state_dict = S3DiffImageEnergy_model.state_dict()  # 完整权重
        S3DiffImageEnergy_model_path = os.path.join(args.output_dir, "S3DiffImageEnergy_model.safetensors")
        save_file(S3DiffImageEnergy_model_state_dict, S3DiffImageEnergy_model_path)
        print(f"自定义辅助模型S3DiffImageEnergy_model （完整）已保存至：{S3DiffImageEnergy_model_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)