#!/usr/bin/env python3
"""
evaluation.py
Person C 的核心工作：評估所有模型的生成結果
"""

import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd

# CLIP
import clip

class Evaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Using device: {self.device}")
        
        # 載入 CLIP
        print(" Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print(" CLIP model loaded!")
        
        self.results = {}
    
    def load_results_json(self, json_path):
        """載入某個人的 results.json"""
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    
    def calculate_clip_score(self, image_paths, prompts):
        """
        計算 CLIP Score (text-image alignment)
        越高越好，表示生成的圖片和 prompt 越匹配
        """
        scores = []
        
        for img_path, prompt in tqdm(zip(image_paths, prompts), 
                                     total=len(image_paths),
                                     desc="  Calculating CLIP scores"):
            try:
                # 載入並預處理圖片
                image = Image.open(img_path).convert("RGB")
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                # 處理文字
                text_input = clip.tokenize([prompt], truncate=True).to(self.device)
                
                with torch.no_grad():
                    # 編碼
                    image_features = self.clip_model.encode_image(image_input)
                    text_features = self.clip_model.encode_text(text_input)
                    
                    # 正規化
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # 計算相似度
                    similarity = (image_features @ text_features.T).item()
                    scores.append(similarity)
                    
            except Exception as e:
                print(f"      Error processing {img_path}: {e}")
                scores.append(0.0)
        
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "scores": scores
        }
    
    def calculate_generation_stats(self, results_list):
        """計算生成時間統計"""
        times = [r["generation_time"] for r in results_list]
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": np.sum(times),
        }
    
    def evaluate_model(self, results_json_path, model_name):
        """評估單一模型"""
        print(f"\n{'='*60}")
        print(f" Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # 載入結果
        results = self.load_results_json(results_json_path)
        
        # 提取圖片路徑和 prompts
        image_paths = [r["image_path"] for r in results]
        prompts = [r["prompt"] for r in results]
        
        print(f"  Images: {len(image_paths)}")
        
        # 計算 CLIP Score
        clip_scores = self.calculate_clip_score(image_paths, prompts)
        
        # 計算生成時間統計
        time_stats = self.calculate_generation_stats(results)
        
        # 整合結果
        eval_results = {
            "model_name": model_name,
            "num_images": len(results),
            "clip_score_mean": clip_scores["mean"],
            "clip_score_std": clip_scores["std"],
            "clip_score_min": clip_scores["min"],
            "clip_score_max": clip_scores["max"],
            **time_stats
        }
        
        print(f"\n  Results:")
        print(f"    CLIP Score: {clip_scores['mean']:.4f} ± {clip_scores['std']:.4f}")
        print(f"    Range: [{clip_scores['min']:.4f}, {clip_scores['max']:.4f}]")
        print(f"    Avg Time: {time_stats['mean_time']:.2f}s")
        print(f"    Total Time: {time_stats['total_time']/60:.1f} min")
        
        return eval_results, clip_scores["scores"]
    
    def create_comparison_table(self, model_configs):
        """
        整合所有模型的評估結果
        
        model_configs = [
            {"name": "FLUX", "results_path": "results/person_a/flux_results.json"},
            ...
        ]
        """
        print("\n" + "="*60)
        print(" Evaluating All Models")
        print("="*60)
        
        all_results = []
        all_clip_scores = {}
        
        for config in model_configs:
            eval_result, clip_scores = self.evaluate_model(
                config["results_path"],
                config["name"]
            )
            all_results.append(eval_result)
            all_clip_scores[config["name"]] = clip_scores
        
        # 建立 DataFrame
        df = pd.DataFrame(all_results)
        df = df.set_index("model_name")
        
        # 計算 Quality/Speed ratio
        df["quality_speed_ratio"] = df["clip_score_mean"] / df["mean_time"]
        
        # 排序（按 CLIP Score）
        df = df.sort_values("clip_score_mean", ascending=False)
        
        # 儲存
        output_dir = Path("evaluation")
        output_dir.mkdir(exist_ok=True)
        
        csv_path = output_dir / "comparison_table.csv"
        df.to_csv(csv_path)
        
        # 同時儲存為 JSON（更好閱讀）
        json_path = output_dir / "comparison_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "models": all_results,
                "clip_scores": {k: [float(v) for v in vals] 
                               for k, vals in all_clip_scores.items()}
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(" Comparison Table")
        print(f"{'='*60}")
        print(df.to_string())
        print(f"{'='*60}")
        
        print(f"\n Results saved to:")
        print(f"    {csv_path}")
        print(f"    {json_path}")
        
        return df, all_clip_scores

def main():
    """主函數：評估所有模型"""
    
    print("\n Evaluation System")
    print("Person C - Gen Comparison Project")
    print("="*60)
    
    # 初始化 evaluator
    evaluator = Evaluator()
    
    # 定義要評估的模型
    model_configs = [
        {
            "name": "FLUX.1-dev",
            "results_path": "results/person_a/flux_results.json"
        },
        {
            "name": "SDXL",
            "results_path": "results/person_a/sdxl_results.json"
        },
        {
            "name": "SD-1.5",
            "results_path": "results/person_b/sd15_results.json"
        },
        {
            "name": "SD-2.1",
            "results_path": "results/person_b/sd21_results.json"
        },
        {
            "name": "ControlNet",
            "results_path": "results/person_c/controlnet_results.json"
        },
    ]
    
    # 檢查檔案是否存在
    print("\n Checking result files...")
    for config in model_configs:
        path = Path(config["results_path"])
        if path.exists():
            print(f"   {config['name']}: {path}")
        else:
            print(f"   {config['name']}: {path} (NOT FOUND)")
    
    print("\n Tip: If files are missing, run generate_mock_results.py first")
    print()
    
    # 評估所有模型
    df, clip_scores = evaluator.create_comparison_table(model_configs)
    
    # 輸出摘要
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    print(f"  Best CLIP Score: {df['clip_score_mean'].idxmax()} ({df['clip_score_mean'].max():.4f})")
    print(f"  Fastest Model: {df['mean_time'].idxmin()} ({df['mean_time'].min():.2f}s)")
    print(f"  Best Quality/Speed: {df['quality_speed_ratio'].idxmax()} ({df['quality_speed_ratio'].max():.4f})")
    print("="*60)
    
    print("\n Evaluation complete!")
    print("\n Next step: Run visualize_results.py")

if __name__ == "__main__":
    main()