#!/usr/bin/env python3
"""
person_c_controlnet.py
Person C - ControlNet 實驗主程式
精簡高效版：今天完成
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
from prompts import PROMPTS

class ControlNetExperiment:
    def __init__(self, output_dir="results/person_c"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = []
        
        print(f"🎮 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    def load_model(self):
        """載入 ControlNet 模型"""
        print("\n" + "="*60)
        print("📥 Loading ControlNet model...")
        print("="*60)
        
        try:
            # 載入 ControlNet
            print("Loading ControlNet Canny...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16,
                cache_dir="./models",
                local_files_only=False,
                use_safetensors=True,
            )
            
            # 載入 SD 1.5 pipeline
            print("Loading Stable Diffusion 1.5 pipeline...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                cache_dir="./models",
                local_files_only=False,
                use_safetensors=True,
            )
            
            # 移到 GPU
            self.pipe = self.pipe.to(self.device)
            
            # 啟用記憶體優化
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_attention_slicing()
            
            print("✅ Model loaded successfully!")
            
        except ImportError as e:
            print(f"❌ Import Error: {e}")
            print("\n🔧 Fixing dependencies...")
            print("Please run:")
            print("  pip install --upgrade transformers==4.30.0")
            print("  pip install --upgrade diffusers==0.21.0")
            print("  pip install --upgrade accelerate==0.20.0")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Tip: Make sure you've installed all dependencies!")
            raise
    
    def get_canny_edge(self, image_path, low_threshold=100, high_threshold=200):
        """從圖片提取 Canny edges"""
        # 讀取圖片
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        
        # 轉灰階
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # 轉成 PIL Image
        canny_image = Image.fromarray(edges)
        
        return canny_image
    
    def generate(self, prompt, control_image, seed=42):
        """使用 ControlNet 生成圖片"""
        generator = torch.Generator(self.device).manual_seed(seed)
        
        start_time = time.time()
        
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                image=control_image,
                num_inference_steps=20,  # 減少步數加快速度
                guidance_scale=7.5,
                generator=generator,
                num_images_per_prompt=1,
            )
        
        gen_time = time.time() - start_time
        
        return output.images[0], gen_time
    
    def run_experiments(self, num_prompts=25):
        """
        執行 ControlNet 實驗
        num_prompts: 要測試的 prompt 數量（預設25個，約1-2小時）
        """
        print("\n" + "="*60)
        print("🚀 Starting ControlNet Experiments")
        print("="*60)
        
        # 載入模型
        self.load_model()
        
        # 取得 control images
        control_dir = Path("control_images/simple_shapes")
        control_images = sorted(list(control_dir.glob("*.png")))
        
        if not control_images:
            print("❌ No control images found!")
            return
        
        print(f"\n📸 Found {len(control_images)} control images")
        print(f"📝 Will test {num_prompts} prompts")
        print(f"⏱️  Estimated time: {num_prompts * 30 / 60:.1f} minutes\n")
        
        # 對每個 control image 測試多個 prompts
        prompts_per_control = num_prompts // len(control_images)
        
        total_generated = 0
        
        for ctrl_idx, ctrl_img_path in enumerate(control_images):
            print(f"\n{'='*60}")
            print(f"🎨 Control Image {ctrl_idx + 1}/{len(control_images)}: {ctrl_img_path.name}")
            print(f"{'='*60}")
            
            # 生成 Canny edge
            print("  Extracting Canny edges...")
            canny_image = self.get_canny_edge(ctrl_img_path)
            canny_save_path = self.output_dir / f"canny_{ctrl_idx:03d}.png"
            canny_image.save(canny_save_path)
            
            # 用這個 control 測試多個 prompts
            start_idx = ctrl_idx * prompts_per_control
            end_idx = min(start_idx + prompts_per_control, len(PROMPTS))
            test_prompts = PROMPTS[start_idx:end_idx]
            
            print(f"  Testing {len(test_prompts)} prompts...")
            
            for prompt_idx, prompt in enumerate(tqdm(test_prompts, desc=f"  {ctrl_img_path.stem}")):
                global_idx = start_idx + prompt_idx
                
                # 生成圖片
                image, gen_time = self.generate(prompt, canny_image, seed=42+global_idx)
                
                # 儲存結果
                save_path = self.output_dir / f"controlnet_{global_idx:03d}.png"
                image.save(save_path)
                
                # 記錄結果
                self.results.append({
                    "model": "ControlNet-Canny",
                    "control_image": str(ctrl_img_path),
                    "canny_image": str(canny_save_path),
                    "prompt": prompt,
                    "prompt_index": global_idx,
                    "generation_time": gen_time,
                    "image_path": str(save_path),
                })
                
                total_generated += 1
        
        # 儲存 results.json
        results_json = self.output_dir / "controlnet_results.json"
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✅ ControlNet 實驗完成！")
        print(f"{'='*60}")
        print(f"📊 總共生成: {total_generated} 張圖片")
        print(f"⏱️  平均生成時間: {np.mean([r['generation_time'] for r in self.results]):.2f}s")
        print(f"💾 結果儲存在: {self.output_dir}")
        print(f"📄 JSON 記錄: {results_json}")
        print(f"{'='*60}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Person C - ControlNet Experiments")
    parser.add_argument("--num_prompts", type=int, default=25,
                       help="Number of prompts to test (default: 25)")
    args = parser.parse_args()
    
    print("\n🎯 Person C - ControlNet Experiment")
    print("Comparative Study of Text-to-Image Generation")
    print("="*60)
    
    experiment = ControlNetExperiment()
    experiment.run_experiments(num_prompts=args.num_prompts)
    
    print("\n🎉 Done! Next step: Run evaluation.py")

if __name__ == "__main__":
    import numpy as np  # for average calculation
    main()