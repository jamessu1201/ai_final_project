# code/analyze_results.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_person_b_results():
    """分析Person B的結果"""
    
    # 讀取結果
    with open("../results/person_b/results.json") as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("="*60)
    print("📊 Person B 結果分析")
    print("="*60)
    
    # 按實驗分組統計
    print("\\n### 各實驗統計 ###")
    for exp in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp]
        print(f"\\n{exp}:")
        print(f"  圖片數量: {len(exp_df)}")
        print(f"  平均生成時間: {exp_df['generation_time'].mean():.2f}s")
        print(f"  平均記憶體使用: {exp_df['peak_memory_gb'].mean():.2f}GB")
    
    # 視覺化：生成時間比較
    plt.figure(figsize=(10, 6))
    df.boxplot(column='generation_time', by='experiment')
    plt.title('Generation Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Experiment')
    plt.tight_layout()
    plt.savefig('../results/person_b/generation_time_comparison.png')
    print("\\n📊 圖表已儲存: generation_time_comparison.png")
    
    # 輸出summary CSV
    summary = df.groupby('experiment').agg({
        'generation_time': ['mean', 'std', 'min', 'max'],
        'peak_memory_gb': ['mean', 'max'],
    }).round(2)
    
    summary.to_csv('../results/person_b/summary.csv')
    print("📄 Summary已儲存: summary.csv")
    
    return df

if __name__ == "__main__":
    df = analyze_person_b_results()