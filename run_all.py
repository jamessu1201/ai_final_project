#!/usr/bin/env python3
"""
run_all.py
一鍵執行整個 Person C 的工作流程
今天完成版
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """執行命令並顯示進度"""
    print("\n" + "="*60)
    print(f" {description}")
    print("="*60)
    print(f"Command: {cmd}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"\n {description} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n {description} failed!")
        print(f"Error: {e}")
        return False

def check_prerequisites():
    """檢查前置條件"""
    print("\n Checking prerequisites...")
    
    # 檢查 prompts.py
    if not Path("prompts.py").exists():
        print(" prompts.py not found!")
        return False
    
    # 檢查 CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA not available, will be slow!")
        else:
            print(f" GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("  PyTorch not installed properly")
    
    print(" Prerequisites OK")
    return True

def main():
    print("\n" + "🎯"*30)
    print("Person C - Complete Pipeline")
    print("Comparative Study of Text-to-Image Generation")
    print("🎯"*30)
    
    # 檢查前置條件
    if not check_prerequisites():
        print("\n Prerequisites check failed!")
        sys.exit(1)
    
    print("\n Pipeline Steps:")
    print("  1. Download models (skip if already done)")
    print("  2. Prepare control images")
    print("  3. Run ControlNet experiments")
    print("  4. Generate mock results (for testing)")
    print("  5. Run evaluation")
    print("  6. Create visualizations")
    print("  7. Organize report materials")
    
    input("\n  Press Enter to start, or Ctrl+C to cancel...")
    
    start_time = time.time()
    
    # Step 1: 下載模型（可選）
    print("\n" + "━"*60)
    print("STEP 1: Download Models")
    print("━"*60)
    
    if Path("models").exists() and any(Path("models").iterdir()):
        print("  Models directory exists, skipping download...")
    else:
        response = input("Download models now? (y/n) [y]: ").strip().lower()
        if response != 'n':
            if not run_command("python download_models.py", 
                             "Step 1: Download Models"):
                print("  Model download failed, but continuing...")
    
    # Step 2: 準備 control images
    print("\n" + "━"*60)
    print("STEP 2: Prepare Control Images")
    print("━"*60)
    
    if not run_command("python prepare_control_images.py << EOF\n3\nEOF",
                      "Step 2: Prepare Control Images"):
        print(" Failed to prepare control images!")
        sys.exit(1)
    
    # Step 3: ControlNet 實驗
    print("\n" + "━"*60)
    print("STEP 3: ControlNet Experiments")
    print("━"*60)
    
    num_prompts = input("Number of prompts to test [25]: ").strip() or "25"
    
    if not run_command(f"python person_c_controlnet.py --num_prompts {num_prompts}",
                      "Step 3: ControlNet Experiments"):
        print(" ControlNet experiments failed!")
        sys.exit(1)
    
    # Step 4: 生成模擬結果
    print("\n" + "━"*60)
    print("STEP 4: Generate Mock Results (for testing)")
    print("━"*60)
    
    if not run_command("python generate_mock_results.py",
                      "Step 4: Generate Mock Results"):
        print(" Failed to generate mock results!")
        sys.exit(1)
    
    # Step 5: 評估
    print("\n" + "━"*60)
    print("STEP 5: Evaluation")
    print("━"*60)
    
    if not run_command("python evaluation.py",
                      "Step 5: Evaluation"):
        print(" Evaluation failed!")
        sys.exit(1)
    
    # Step 6: 視覺化
    print("\n" + "━"*60)
    print("STEP 6: Visualizations")
    print("━"*60)
    
    if not run_command("python visualize_results.py",
                      "Step 6: Visualizations"):
        print(" Visualization failed!")
        sys.exit(1)
    
    # Step 7: 整理報告材料
    print("\n" + "━"*60)
    print("STEP 7: Organize Report Materials")
    print("━"*60)
    
    report_dir = Path("report_materials")
    report_dir.mkdir(exist_ok=True)
    
    # 複製檔案
    import shutil
    
    print("   Copying files...")
    
    # Table
    if Path("evaluation/comparison_table.csv").exists():
        shutil.copy("evaluation/comparison_table.csv",
                   report_dir / "table1_model_comparison.csv")
        print("     comparison_table.csv")
    
    # Charts
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    chart_files = [
        "clip_score_comparison.png",
        "generation_time_comparison.png",
        "quality_vs_speed.png",
        "clip_score_distribution.png",
    ]
    
    for chart in chart_files:
        src = Path("visualizations") / chart
        if src.exists():
            shutil.copy(src, figures_dir / chart)
            print(f"     {chart}")
    
    # Visual comparisons
    comparisons_dir = figures_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)
    
    for i in range(5):
        src = Path("visualizations") / f"comparison_{i:03d}.png"
        if src.exists():
            shutil.copy(src, comparisons_dir / f"visual_comparison_{i+1}.png")
    
    print("   Report materials organized!")
    
    # 完成
    total_time = time.time() - start_time
    
    print("\n" + "🎉"*30)
    print(" PIPELINE COMPLETE!")
    print("🎉"*30)
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    
    print("\n Generated files:")
    print("   evaluation/comparison_table.csv")
    print("   evaluation/comparison_results.json")
    print("   visualizations/*.png (4 charts + 5 comparisons)")
    print("   report_materials/ (organized for report)")
    
    print("\n What's next:")
    print("  1. Review the results in evaluation/")
    print("  2. Check visualizations/")
    print("  3. Use report_materials/ for your report")
    print("  4. Share prompts.py with Person A and B")
    print("  5. Replace mock results with real results when available")
    
    print("\n Tips:")
    print("  - Results are in evaluation/comparison_table.csv")
    print("  - Charts are in visualizations/")
    print("  - Report materials in report_materials/")
    
    print("\n🎓 Good luck with your project!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user")
        sys.exit(1)