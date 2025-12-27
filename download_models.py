#!/usr/bin/env python3
"""
download_models.py
下載 ControlNet 和 Stable Diffusion 模型
Person C - 可以 overnight 執行
"""

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from pathlib import Path
import sys

def check_gpu():
    """檢查 GPU 可用性"""
    print("="*60)
    print(" Checking GPU...")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f" CUDA available: True")
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(" CUDA not available!")
        print("  Models will download but you won't be able to run them without GPU")
    print()

def download_controlnet():
    """下載 ControlNet Canny 模型"""
    print("="*60)
    print(" Downloading ControlNet Canny model...")
    print("="*60)
    print(" This may take 10-20 minutes depending on your connection")
    print()
    
    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16,
            cache_dir="./models"  # 儲存在本地
        )
        print(" ControlNet Canny downloaded successfully!")
        return True
    except Exception as e:
        print(f" Error downloading ControlNet: {e}")
        return False

def download_sd15():
    """下載 Stable Diffusion 1.5 base model"""
    print("\n" + "="*60)
    print(" Downloading Stable Diffusion 1.5...")
    print("="*60)
    print(" This may take 10-20 minutes depending on your connection")
    print()
    
    try:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            cache_dir="./models",
            safety_checker=None,  # 節省下載時間
        )
        print(" Stable Diffusion 1.5 downloaded successfully!")
        return True
    except Exception as e:
        print(f" Error downloading SD 1.5: {e}")
        return False

def main():
    print("\n ControlNet Model Downloader")
    print("Person C - Gen Comparison Project")
    print("="*60)
    
    # 建立 models 目錄
    Path("./models").mkdir(exist_ok=True)
    
    # 檢查 GPU
    check_gpu()
    
    # 下載模型
    print("\n Starting downloads...")
    print(" Tip: This script can run overnight. Feel free to do other things!")
    print()
    
    success_count = 0
    
    # 下載 ControlNet
    if download_controlnet():
        success_count += 1
    
    # 下載 SD 1.5
    if download_sd15():
        success_count += 1
    
    # 總結
    print("\n" + "="*60)
    print(" Download Summary")
    print("="*60)
    print(f" Successfully downloaded: {success_count}/2 models")
    
    if success_count == 2:
        print("\n All models downloaded successfully!")
        print("\n Next steps:")
        print("  1. Prepare control images (prepare_control_images.py)")
        print("  2. Run ControlNet experiments (person_c_controlnet.py)")
    else:
        print("\n  Some downloads failed. Please check your internet connection and try again.")
    
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Download interrupted by user")
        print(" You can run this script again to resume downloads")
        sys.exit(1)