# usr/bin/env python3
"""
Script running PhotoMaker pipeline for stylizing images
when using change the input_root and output_root variables

code is inspired by https://github.com/TencentARC/PhotoMaker app_v2.py
file: run_photomaker.py
project: KNN Face Recognition
author: Tereza Magerkova, xmager00
"""
import os
import torch
import numpy as np
from tqdm import tqdm

import onnxruntime as ort
ort.set_default_logger_severity(3) # silence provides logs when executing

from diffusers import EulerDiscreteScheduler, T2IAdapter
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
# from PhotoMaker repo
from photomaker import FaceAnalysis2, analyze_faces
from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline

# select device and dtype
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

print(f"[INFO] Using device={device}, dtype={dtype}")

# face detector from PhotoMaker
providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if device.type == "cuda"
             else ["CPUExecutionProvider"])
face_detector = FaceAnalysis2(
    providers=providers,
    allowed_modules=['detection', 'recognition']
)
face_detector.prepare(ctx_id=0, det_size=(640, 640))

# checkpoint and model
ckpt = hf_hub_download(
    repo_id="TencentARC/PhotoMaker-V2",
    filename="photomaker-v2.bin",
    repo_type="model"
)
base_model = 'SG161222/RealVisXL_V4.0'
adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0",
    torch_dtype=dtype,
    variant="fp16" if device.type == "cuda" else None
).to(device)

pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
    base_model,
    adapter=adapter,
    torch_dtype=dtype,
    use_safetensors=True,
    variant="fp16" if device.type == "cuda" else None
).to(device)

pipe.load_photomaker_adapter(
    os.path.dirname(ckpt),
    weight_name=os.path.basename(ckpt),
    trigger_word="img",
    pm_version="v2"
)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.fuse_lora()
pipe.to(device)

# parameters for stylization
INPUT_ROOT  = "../casia_images"
OUTPUT_ROOT =   "./stylized_images"

NEG_PROMPT = "nude, modern clothing, makeup, watermark, text, digital artifacts, oversaturation, cartoon, bad anatomy, artificial lighting"
STEPS       = 50
STYLE_RATIO = 15
GUIDANCE    = 5.0
WIDTH, HEIGHT = 1024, 1024
MAX_SEED    = np.iinfo(np.int32).max

# use multiple prompts to change image style
with open("../prompts.txt", 'r') as f:
    prompts = [line.strip() for line in f if line.strip()]

merge_step = min(int(STYLE_RATIO/100 * STEPS), STEPS)

# loop over batch
print("Starting stylization")
for identity in sorted(os.listdir(INPUT_ROOT)):
    print(f"Identity: {identity}")
    src_dir = os.path.join(INPUT_ROOT, identity)
    dst_dir = os.path.join(OUTPUT_ROOT, identity)
    if not os.path.isdir(src_dir):
        continue
    os.makedirs(dst_dir, exist_ok=True)

    for idx, img_name in enumerate(sorted(os.listdir(src_dir))):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        in_path  = os.path.join(src_dir, img_name)
        out_path = os.path.join(dst_dir, img_name)
        print(f"Image: {img_name}\t")

        pil_img = load_image(in_path)
        arr     = np.array(pil_img)[..., ::-1] # RGB to BGR for the InsightFace
        faces   = analyze_faces(face_detector, arr)
        if not faces:
            print(f"No face found in {in_path}")
            continue
        emb = torch.from_numpy(faces[0].embedding).to(device) # face to tensor
        seed = np.random.randint(0, MAX_SEED)
        prompt = prompts[idx]

        gen = torch.Generator(device=device).manual_seed(seed) # "stable" randomness for diffusion process

        out = pipe(
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            input_id_images=[pil_img],
            id_embeds=emb,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            num_images_per_prompt=1,
            start_merge_step=merge_step,
            generator=gen,
            width=WIDTH,
            height=HEIGHT
        ).images[0] # take only one image

        out.save(out_path)

print("Done")
