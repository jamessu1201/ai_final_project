#!/usr/bin/env python3
"""
visualize_results.py
建立漂亮的視覺對比圖，用於報告和投影片
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from prompts import PROMPTS

# 設定中文字體（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_evaluation_results(self):
        """載入評估結果"""
        csv_path = Path("evaluation/comparison_table.csv")
        json_path = Path("evaluation/comparison_results.json")
        
        df = pd.read_csv(csv_path, index_col=0)
        
        with open(json_path, encoding="utf-8") as f:
            results_json = json.load(f)
        
        return df, results_json
    
    def plot_clip_scores(self, df):
        """繪製 CLIP Score 比較圖"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = df.index
        scores = df["clip_score_mean"]
        errors = df["clip_score_std"]
        
        bars = ax.bar(models, scores, yerr=errors, capsize=5, 
                     color=sns.color_palette("husl", len(models)))
        
        ax.set_ylabel("CLIP Score", fontsize=12, fontweight='bold')
        ax.set_title("Text-Image Alignment Comparison", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0, max(scores) * 1.2])
        
        # 在每個 bar 上標註數值
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / "clip_score_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   {save_path}")
        plt.close()
    
    def plot_generation_time(self, df):
        """繪製生成時間比較圖"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = df.index
        times = df["mean_time"]
        errors = df["std_time"]
        
        bars = ax.bar(models, times, yerr=errors, capsize=5,
                     color=sns.color_palette("muted", len(models)))
        
        ax.set_ylabel("Generation Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("Generation Speed Comparison",
                    fontsize=14, fontweight='bold', pad=20)
        
        # 標註數值
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.2f}s',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / "generation_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   {save_path}")
        plt.close()
    
    def plot_quality_vs_speed(self, df):
        """繪製 Quality vs Speed 散點圖"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("husl", len(df))
        
        for i, (model, row) in enumerate(df.iterrows()):
            ax.scatter(row["mean_time"], row["clip_score_mean"],
                      s=300, alpha=0.6, color=colors[i], label=model,
                      edgecolors='black', linewidth=1.5)
            
            # 標註模型名稱
            ax.annotate(model,
                       (row["mean_time"], row["clip_score_mean"]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor=colors[i], alpha=0.3))
        
        ax.set_xlabel("Generation Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_ylabel("CLIP Score", fontsize=12, fontweight='bold')
        ax.set_title("Quality vs Speed Trade-off",
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "quality_vs_speed.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   {save_path}")
        plt.close()
    
    def plot_clip_distribution(self, clip_scores_dict):
        """繪製 CLIP Score 分佈 (Box plot)"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(clip_scores_dict.keys())
        data = [clip_scores_dict[model] for model in models]
        
        bp = ax.boxplot(data, labels=models, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # 為每個 box 設定不同顏色
        colors = sns.color_palette("husl", len(models))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel("CLIP Score", fontsize=12, fontweight='bold')
        ax.set_title("CLIP Score Distribution",
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / "clip_score_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   {save_path}")
        plt.close()
    
    def create_comparison_grid(self, prompt_id):
        """
        為單一 prompt 建立所有模型的對比圖
        
        Layout:
        [Prompt Text]
        [FLUX] [SDXL] [SD-1.5] [SD-2.1] [ControlNet]
        """
        prompt = PROMPTS[prompt_id]
        
        # 載入所有模型的圖片
        image_paths = {
            "FLUX": f"results/person_a/flux_{prompt_id:03d}.png",
            "SDXL": f"results/person_a/sdxl_{prompt_id:03d}.png",
            "SD-1.5": f"results/person_b/sd15_baseline_{prompt_id:03d}.png",
            "SD-2.1": f"results/person_b/sd21_baseline_{prompt_id:03d}.png",
            "ControlNet": f"results/person_c/controlnet_{prompt_id:03d}.png",
        }
        
        images = {}
        for model, path in image_paths.items():
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((256, 256))
                images[model] = img
            except:
                # 如果圖片不存在，建立灰色佔位符
                images[model] = Image.new("RGB", (256, 256), color='gray')
        
        # 建立大畫布
        canvas_width = 256 * 5 + 40 * 6
        canvas_height = 256 + 80 + 60
        
        canvas = Image.new("RGB", (canvas_width, canvas_height), color='white')
        draw = ImageDraw.Draw(canvas)
        
        # 字體
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_label = font_title
        
        # 寫 prompt
        prompt_short = prompt if len(prompt) < 100 else prompt[:97] + "..."
        draw.text((20, 15), f'Prompt #{prompt_id}: "{prompt_short}"',
                 fill='black', font=font_title)
        
        # 貼上圖片
        x_offset = 40
        y_offset = 60
        
        for model, img in images.items():
            # 貼圖
            canvas.paste(img, (x_offset, y_offset))
            
            # 寫模型名稱
            bbox = draw.textbbox((0, 0), model, font=font_label)
            text_width = bbox[2] - bbox[0]
            text_x = x_offset + (256 - text_width) // 2
            text_y = y_offset + 256 + 10
            
            draw.text((text_x, text_y), model, fill='black', font=font_label)
            
            x_offset += 256 + 40
        
        # 儲存
        save_path = self.output_dir / f"comparison_{prompt_id:03d}.png"
        canvas.save(save_path)
        
        return save_path
    
    def create_all_visualizations(self):
        """建立所有視覺化"""
        print("\n" + "="*60)
        print(" Creating Visualizations")
        print("="*60)
        
        # 載入評估結果
        print("\n Loading evaluation results...")
        df, results_json = self.load_evaluation_results()
        clip_scores = results_json["clip_scores"]
        
        # 繪製圖表
        print("\n Generating charts...")
        self.plot_clip_scores(df)
        self.plot_generation_time(df)
        self.plot_quality_vs_speed(df)
        self.plot_clip_distribution(clip_scores)
        
        # 建立視覺對比圖（前 5 個 prompts）
        print("\n Generating visual comparisons...")
        for i in range(min(5, len(PROMPTS))):
            save_path = self.create_comparison_grid(i)
            print(f"   {save_path}")
        
        print("\n" + "="*60)
        print(" All visualizations created!")
        print("="*60)
        print(f"\n Saved to: {self.output_dir}/")
        print("\n Next step: Organize report materials")

def main():
    print("\n Visualization System")
    print("Person C - Gen Comparison Project")
    print("="*60)
    
    visualizer = Visualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()