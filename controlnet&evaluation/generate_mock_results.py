#!/usr/bin/env python3
"""
generate_mock_results.py
生成模擬的 Person A 和 B 的結果
這樣 Person C 可以獨立測試評估系統
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from prompts import PROMPTS

def create_dummy_image(prompt, model_name, output_path, seed=42):
    """建立一個帶有文字的假圖片（用於測試）"""
    np.random.seed(seed)
    
    # 建立隨機背景
    img = Image.new('RGB', (512, 512), 
                    color=(np.random.randint(100, 255),
                          np.random.randint(100, 255),
                          np.random.randint(100, 255)))
    
    draw = ImageDraw.Draw(img)
    
    # 寫上模型名稱和 prompt 片段
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = font_large
    
    # 模型名稱
    draw.text((20, 20), model_name, fill='white', font=font_large)
    
    # Prompt (縮短)
    prompt_short = prompt if len(prompt) < 40 else prompt[:37] + "..."
    draw.text((20, 60), f'"{prompt_short}"', fill='white', font=font_small)
    
    # 加一些隨機形狀讓圖片看起來不同
    for _ in range(5):
        x1 = np.random.randint(0, 400)
        y1 = np.random.randint(100, 400)
        x2 = x1 + np.random.randint(50, 100)
        y2 = y1 + np.random.randint(50, 100)
        color = (np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    img.save(output_path)
    return output_path

def generate_person_a_results(num_prompts=25):
    """生成 Person A 的模擬結果 (FLUX 和 SDXL)"""
    print("\n Generating Person A mock results...")
    print("="*60)
    
    output_dir = Path("results/person_a")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FLUX 結果
    flux_results = []
    print("  Creating FLUX results...")
    for i in range(num_prompts):
        prompt = PROMPTS[i]
        img_path = output_dir / f"flux_{i:03d}.png"
        create_dummy_image(prompt, "FLUX.1-dev", img_path, seed=1000+i)
        
        flux_results.append({
            "model": "FLUX.1-dev",
            "prompt": prompt,
            "prompt_index": i,
            "generation_time": np.random.uniform(3.5, 5.5),  # FLUX 比較慢
            "image_path": str(img_path),
        })
    
    with open(output_dir / "flux_results.json", "w", encoding="utf-8") as f:
        json.dump(flux_results, f, indent=2, ensure_ascii=False)
    
    # SDXL 結果
    sdxl_results = []
    print("  Creating SDXL results...")
    for i in range(num_prompts):
        prompt = PROMPTS[i]
        img_path = output_dir / f"sdxl_{i:03d}.png"
        create_dummy_image(prompt, "SDXL", img_path, seed=2000+i)
        
        sdxl_results.append({
            "model": "SDXL",
            "prompt": prompt,
            "prompt_index": i,
            "generation_time": np.random.uniform(2.5, 4.0),  # SDXL 中等速度
            "image_path": str(img_path),
        })
    
    with open(output_dir / "sdxl_results.json", "w", encoding="utf-8") as f:
        json.dump(sdxl_results, f, indent=2, ensure_ascii=False)
    
    print(f" Person A results created: {num_prompts} FLUX + {num_prompts} SDXL images")

def generate_person_b_results(num_prompts=25):
    """生成 Person B 的模擬結果 (SD 1.5 和 SD 2.1)"""
    print("\n Generating Person B mock results...")
    print("="*60)
    
    output_dir = Path("results/person_b")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SD 1.5 結果
    sd15_results = []
    print("  Creating SD 1.5 results...")
    for i in range(num_prompts):
        prompt = PROMPTS[i]
        img_path = output_dir / f"sd15_baseline_{i:03d}.png"
        create_dummy_image(prompt, "SD-1.5", img_path, seed=3000+i)
        
        sd15_results.append({
            "model": "SD-1.5",
            "prompt": prompt,
            "prompt_index": i,
            "generation_time": np.random.uniform(1.5, 2.5),  # SD 1.5 最快
            "image_path": str(img_path),
        })
    
    with open(output_dir / "sd15_results.json", "w", encoding="utf-8") as f:
        json.dump(sd15_results, f, indent=2, ensure_ascii=False)
    
    # SD 2.1 結果
    sd21_results = []
    print("  Creating SD 2.1 results...")
    for i in range(num_prompts):
        prompt = PROMPTS[i]
        img_path = output_dir / f"sd21_{i:03d}.png"
        create_dummy_image(prompt, "SD-2.1", img_path, seed=4000+i)
        
        sd21_results.append({
            "model": "SD-2.1",
            "prompt": prompt,
            "prompt_index": i,
            "generation_time": np.random.uniform(1.8, 2.8),  # SD 2.1 稍慢
            "image_path": str(img_path),
        })
    
    with open(output_dir / "sd21_results.json", "w", encoding="utf-8") as f:
        json.dump(sd21_results, f, indent=2, ensure_ascii=False)
    
    print(f" Person B results created: {num_prompts} SD-1.5 + {num_prompts} SD-2.1 images")

def main():
    print("\n Mock Results Generator")
    print("Person C - Gen Comparison Project")
    print("="*60)
    print("\n 這會建立模擬的 Person A 和 B 的結果")
    print("   這樣你可以獨立測試評估系統")
    print()
    
    num_prompts = 25  # 與 ControlNet 實驗數量一致
    
    generate_person_a_results(num_prompts)
    generate_person_b_results(num_prompts)
    
    print("\n" + "="*60)
    print(" All mock results generated!")
    print("="*60)
    print("\n Generated files:")
    print("  results/person_a/flux_results.json")
    print("  results/person_a/sdxl_results.json")
    print("  results/person_b/sd15_results.json")
    print("  results/person_b/sd21_results.json")
    print("\n Note: 這些是測試用的假資料")
    print("   等 Person A 和 B 完成後，用真實資料替換")
    print("\n Next step: Run evaluation.py")

if __name__ == "__main__":
    main()