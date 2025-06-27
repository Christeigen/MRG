import pandas as pd
import numpy as np
import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
from PIL import Image
import cv2
import peft
import json
from peft import get_peft_model, LoraConfig, TaskType
from configs.config import EncoderConfig

class CLIPWithLoRA(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model and preprocess
        self.clip_model, self.preprocess = clip.load(config.clip_model_name, device=self.device)

        # Inject LoRA into target modules
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["out_proj"]
        )

        use_lora = config.enable_lora
        if use_lora == True:
            self.clip_model = get_peft_model(self.clip_model, lora_config).to(self.device)
        else:
            pass

    def get_patch_tokens(self, clip_model, pixel_values):
        x = clip_model.conv1(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [clip_model.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x],
            dim=1,
        )

        x = x + clip_model.positional_embedding
        x = clip_model.ln_pre(x)

        x = x.permute(1, 0, 2)  # for transformer: [S, B, D]
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # back to [B, S, D]

        x = clip_model.ln_post(x)  # [B, 50, 768]

        return x

    def forward(self, images):
        # images: [B, 3, H, W]
        image_features = self.get_patch_tokens(self.clip_model.visual, images)
        # last_hidden_state = image_features["last_hidden_state"]  # [B, Num_Patches, Embed_Dim]
        return image_features

    def get_preprocess(self):
        return self.preprocess