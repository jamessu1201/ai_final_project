# analyze_person_a.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

class PersonA_Analyzer:
    def __init__(self):
        with open("..//results.json") as f:
            self.data = json.load(f)
    
    def plot_generation_time_comparison(self):
        """比較FLUX vs SDXL的生成時間"""
        flux_times = [x['generation_time'] for x in self.data 
                      if x['model'] == 'FLUX.1-dev']
        sdxl_times = [x['generation_time'] for x in self.data 
                      if x['model'] == 'SDXL']
        
        plt.figure(figsize=(10, 6))
        plt.boxplot([flux_times, sdxl_times], labels=['FLUX.1-dev', 'SDXL'])
        plt.ylabel('Generation Time (seconds)')
        plt.title('Generation Time Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig('generation_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"FLUX平均: {np.mean(flux_times):.2f}s ± {np.std(flux_times):.2f}s")
        print(f"SDXL平均: {np.mean(sdxl_times):.2f}s ± {np.std(sdxl_times):.2f}s")
    
    def create_visual_comparison_grid(self, num_samples=5):
        """創建視覺對比圖（FLUX vs SDXL）"""
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            # Prompt
            prompt = self.data[i]['prompt']
            axes[i, 0].text(0.5, 0.5, prompt, 
                           ha='center', va='center', wrap=True, fontsize=10)
            axes[i, 0].axis('off')
            axes[i, 0].set_title('Prompt', fontsize=12, fontweight='bold')
            
            # FLUX
            flux_img = Image.open(f"../flux_{i:03d}.png")
            axes[i, 1].imshow(flux_img)
            axes[i, 1].axis('off')
            axes[i, 1].set_title('FLUX.1-dev', fontsize=12, fontweight='bold')
            
            # SDXL
            sdxl_img = Image.open(f"../sdxl_{i:03d}.png")
            axes[i, 2].imshow(sdxl_img)
            axes[i, 2].axis('off')
            axes[i, 2].set_title('SDXL', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visual_comparison_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("視覺對比圖已儲存！")
    
    def analyze_by_category(self):
        """按照prompt類別分析生成時間"""
        from prompts import CATEGORIES
        
        category_times = {cat: {'flux': [], 'sdxl': []} 
                         for cat in CATEGORIES.keys()}
        
        for entry in self.data:
            prompt_id = entry['prompt_id']
            model = 'flux' if entry['model'] == 'FLUX.1-dev' else 'sdxl'
            gen_time = entry['generation_time']
            
            # 找到這個prompt屬於哪個category
            for cat_name, prompts in CATEGORIES.items():
                if entry['prompt'] in prompts:
                    category_times[cat_name][model].append(gen_time)
                    break
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(12, 6))
        categories = list(category_times.keys())
        x = np.arange(len(categories))
        width = 0.35
        
        flux_means = [np.mean(category_times[cat]['flux']) for cat in categories]
        sdxl_means = [np.mean(category_times[cat]['sdxl']) for cat in categories]
        
        ax.bar(x - width/2, flux_means, width, label='FLUX.1-dev', alpha=0.8)
        ax.bar(x + width/2, sdxl_means, width, label='SDXL', alpha=0.8)
        
        ax.set_xlabel('Prompt Category')
        ax.set_ylabel('Average Generation Time (s)')
        ax.set_title('Generation Time by Prompt Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("類別分析圖已儲存！")

if __name__ == "__main__":
    analyzer = PersonA_Analyzer()
    
    print("=== 分析Person A的結果 ===")
    print("\n1. 生成時間比較...")
    analyzer.plot_generation_time_comparison()
    
    print("\n2. 創建視覺對比圖...")
    analyzer.create_visual_comparison_grid(num_samples=10)
    
    print("\n3. 按類別分析...")
    analyzer.analyze_by_category()
    
    print("\n✅ 所有分析完成！")