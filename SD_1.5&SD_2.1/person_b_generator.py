# code/person_b_generator.py
import torch
from diffusers import StableDiffusionPipeline
from prompts import PROMPTS, NEGATIVE_PROMPT
import time
import json
from pathlib import Path
from datetime import datetime
import traceback

class PersonB_Generator:
    def __init__(self, output_dir="../results/person_b"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda"
        self.results = []
        
        # 記錄GPU資訊
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
    def load_sd15(self):
        """載入SD 1.5"""
        print("\\n📥 Loading Stable Diffusion 1.5...")
        self.sd15_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,  # 關閉safety checker加速
        ).to(self.device)
        
        # 啟用記憶體優化
        self.sd15_pipe.enable_attention_slicing()
        print("✅ SD 1.5 loaded!")
        
    def load_sd21(self):
        """載入SD 2.1"""
        print("\\n📥 Loading Stable Diffusion 2.1 (Manojb Version)...")
        self.sd21_pipe = StableDiffusionPipeline.from_pretrained(
            "Manojb/stable-diffusion-2-1-base",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)
        
        self.sd21_pipe.enable_attention_slicing()
        print("✅ SD 2.1 loaded!")
    
    def generate_image(self, pipe, prompt, negative_prompt=None, seed=42):
        """生成單張圖片"""
        generator = torch.Generator(self.device).manual_seed(seed)
        
        # 記錄開始時間和記憶體
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=7.5,
                num_inference_steps=30,
                generator=generator,
                height=512,
                width=512,
            ).images[0]
            
            gen_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return image, gen_time, peak_memory, True
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            traceback.print_exc()
            return None, 0, 0, False
    
    def experiment_1_sd15_baseline(self):
        """實驗1: SD 1.5 基礎版本"""
        print("\\n" + "="*60)
        print("🧪 實驗1: SD 1.5 Baseline")
        print("="*60)
        
        self.load_sd15()
        
        for i, prompt in enumerate(PROMPTS):
            print(f"\\n[{i+1}/50] Generating with SD 1.5...")
            print(f"Prompt: {prompt[:50]}...")
            
            image, gen_time, peak_mem, success = self.generate_image(
                self.sd15_pipe, prompt
            )
            
            if not success:
                continue
            
            # 儲存圖片
            save_path = self.output_dir / f"sd15_baseline_{i:03d}.png"
            image.save(save_path)
            
            # 記錄結果
            result = {
                "experiment": "sd15_baseline",
                "model": "SD-1.5",
                "prompt_id": i,
                "prompt": prompt,
                "negative_prompt": None,
                "generation_time": round(gen_time, 2),
                "peak_memory_gb": round(peak_mem, 2),
                "image_path": str(save_path),
                "timestamp": datetime.now().isoformat(),
            }
            self.results.append(result)
            
            print(f"⏱️  Time: {gen_time:.2f}s | 💾 Memory: {peak_mem:.2f}GB")
            
            # 每10張儲存一次（避免crash損失資料）
            if (i + 1) % 10 == 0:
                self.save_results()
                print(f"💾 Progress saved! ({i+1}/50)")
        
        print("\\n✅ 實驗1完成！")
    
    def experiment_2_sd15_negative(self):
        """實驗2: SD 1.5 + Negative Prompt"""
        print("\\n" + "="*60)
        print("🧪 實驗2: SD 1.5 + Negative Prompt")
        print("="*60)
        
        # SD 1.5已經載入，直接使用
        
        for i, prompt in enumerate(PROMPTS):
            print(f"\\n[{i+1}/50] Generating with SD 1.5 + Negative...")
            print(f"Prompt: {prompt[:50]}...")
            
            image, gen_time, peak_mem, success = self.generate_image(
                self.sd15_pipe, prompt, negative_prompt=NEGATIVE_PROMPT
            )
            
            if not success:
                continue
            
            save_path = self.output_dir / f"sd15_negative_{i:03d}.png"
            image.save(save_path)
            
            result = {
                "experiment": "sd15_negative",
                "model": "SD-1.5",
                "prompt_id": i,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "generation_time": round(gen_time, 2),
                "peak_memory_gb": round(peak_mem, 2),
                "image_path": str(save_path),
                "timestamp": datetime.now().isoformat(),
            }
            self.results.append(result)
            
            print(f"⏱️  Time: {gen_time:.2f}s | 💾 Memory: {peak_mem:.2f}GB")
            
            if (i + 1) % 10 == 0:
                self.save_results()
                print(f"💾 Progress saved! ({i+1}/50)")
        
        print("\\n✅ 實驗2完成！")
        
        # 釋放SD 1.5記憶體
        del self.sd15_pipe
        torch.cuda.empty_cache()
        print("🗑️  SD 1.5 unloaded from memory")
    
    def experiment_3_sd21_baseline(self):
        """實驗3: SD 2.1 基礎版本"""
        print("\\n" + "="*60)
        print("🧪 實驗3: SD 2.1 Baseline")
        print("="*60)
        
        self.load_sd21()
        
        for i, prompt in enumerate(PROMPTS):
            print(f"\\n[{i+1}/50] Generating with SD 2.1...")
            print(f"Prompt: {prompt[:50]}...")
            
            image, gen_time, peak_mem, success = self.generate_image(
                self.sd21_pipe, prompt
            )
            
            if not success:
                continue
            
            save_path = self.output_dir / f"sd21_baseline_{i:03d}.png"
            image.save(save_path)
            
            result = {
                "experiment": "sd21_baseline",
                "model": "SD-2.1",
                "prompt_id": i,
                "prompt": prompt,
                "negative_prompt": None,
                "generation_time": round(gen_time, 2),
                "peak_memory_gb": round(peak_mem, 2),
                "image_path": str(save_path),
                "timestamp": datetime.now().isoformat(),
            }
            self.results.append(result)
            
            print(f"⏱️  Time: {gen_time:.2f}s | 💾 Memory: {peak_mem:.2f}GB")
            
            if (i + 1) % 10 == 0:
                self.save_results()
                print(f"💾 Progress saved! ({i+1}/50)")
        
        print("\\n✅ 實驗3完成！")
        
        del self.sd21_pipe
        torch.cuda.empty_cache()
    
    def save_results(self):
        """儲存結果到JSON"""
        with open(self.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def run_all_experiments(self):
        """執行所有實驗"""
        start_time = time.time()
        
        print("\\n" + "="*60)
        print("🚀 Person B - SD Series Comparison")
        print("="*60)
        
        try:
            # 實驗1: SD 1.5 baseline
            self.experiment_1_sd15_baseline()
            
            # 實驗2: SD 1.5 + negative prompt
            self.experiment_2_sd15_negative()
            
            # 實驗3: SD 2.1 baseline
            self.experiment_3_sd21_baseline()
            
            # 最終儲存
            self.save_results()
            
            total_time = time.time() - start_time
            print("\\n" + "="*60)
            print(f"🎉 所有實驗完成！")
            print(f"⏱️  總耗時: {total_time/3600:.2f} 小時")
            print(f"📊 總共生成: {len(self.results)} 張圖片")
            print(f"💾 結果儲存在: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"\\n❌ 發生錯誤: {e}")
            traceback.print_exc()
            self.save_results()  # 即使出錯也儲存已完成的部分

if __name__ == "__main__":
    generator = PersonB_Generator()
    generator.run_all_experiments()