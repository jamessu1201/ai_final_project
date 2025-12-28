# Person A - FLUX vs SDXL 實驗說明

## 🎯 負責任務
比較 FLUX.1-dev 和 SDXL 兩個模型的生成品質與速度

---

## 🖥️ 實驗環境

**硬體配置：**
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- OS: Ubuntu 24

**軟體需求：**
```bash
# 建立環境
conda create -n gen_compare python=3.10
conda activate gen_compare

# 安裝套件
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors
pip install pillow numpy matplotlib pandas
```

---

## 📋 實驗設定

### FLUX.1-dev
- 模型: `black-forest-labs/FLUX.1-dev`
- 推論步數: 28 steps
- Guidance scale: 3.5
- 解析度: 1024×1024
- 資料類型: bfloat16

### SDXL
- 模型: `stabilityai/stable-diffusion-xl-base-1.0`
- 推論步數: 30 steps
- Guidance scale: 7.5
- 解析度: 1024×1024
- 資料類型: float16

### 測試集
- **總數**: 50個prompts
- **分類**: 5個類別，每類10個prompts
  1. 簡單物體 (Simple Objects)
  2. 動物與生物 (Animals & Creatures)
  3. 角色與肖像 (Characters & Portraits)
  4. 場景與風景 (Scenes & Landscapes)
  5. 藝術風格 (Artistic Styles)

---

## 🚀 執行步驟

### 1. 生成圖片
```bash
# 下載並執行flux_sdxl的程式
python flux_sdxl.py
```

程式會自動：
- 載入FLUX.1-dev模型，生成50張圖片
- 載入SDXL模型，生成50張圖片
- 記錄每張圖片的生成時間
- 將結果存到 `results/person_a/` 目錄

**預計時間：**
- FLUX: ~10-11分鐘
- SDXL: ~3分鐘
- 總共: ~13-14分鐘

### 2. 產出檔案
執行完成後會產生：
```
results/person_a/
├── flux_000.png ~ flux_049.png    (50張FLUX生成的圖)
├── sdxl_000.png ~ sdxl_049.png    (50張SDXL生成的圖)
└── results.json                   (包含所有metadata)
```

---

## 🔧 故障排除

**如果遇到GPU記憶體不足：**
```python
# 可以降低batch size或使用更小的模型
# FLUX需要約20-22GB VRAM
# SDXL需要約8-10GB VRAM
```

**如果模型下載很慢：**
```bash
# 可以先手動下載模型
huggingface-cli download black-forest-labs/FLUX.1-dev
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
```
