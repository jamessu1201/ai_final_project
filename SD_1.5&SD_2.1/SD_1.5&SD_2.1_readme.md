## 🎯 負責任務
本專案用於評估與比較 Stable Diffusion 1.5 與 Stable Diffusion 2.1 在圖像生成任務中的效能表現。實驗重點在於測量不同模型版本與配置（如 Negative Prompt 的加入）對 生成時間 與 顯示記憶體 (VRAM) 消耗 的影響。
🧪 實驗設計本實驗透過 prompts.py 定義了 50 組涵蓋五大類別（物體、動物、人像、場景、藝術風格）的提示詞，並分為三個實驗組別：實驗編號實驗名稱模型版本配置說明Exp 1SD15 BaselineSD v1.5基礎生成，無負向提示詞Exp 2SD15 NegativeSD v1.5加入 NEGATIVE\_PROMPT 進行生成Exp 3SD21 BaselineSD v2.1使用 2.1 Base 模型進行效能對照📂 目錄結構Plaintext.

├── code/

│   ├── prompts.py           # 存放 50 組測試 Prompt 與 Negative Prompt

│   ├── person\_b\_generator.py # 核心執行腳本，負責模型載入與圖片生成

│   └── analyze\_results.py   # 數據分析腳本，產出 CSV 報表與統計圖

└── results/

└── person\_b/            # 實驗結果輸出路徑 (自動建立)

├── \*.png            # 生成的實驗圖片

├── results.json     # 原始數據紀錄

├── summary.csv      # 統計彙整表 (Mean/Std/Min/Max)

└── generation\_time\_comparison.png # 生成時間分佈盒鬚圖
---

## 🖥️ 實驗環境

**軟體配置：**
⚙️ 安裝與準備環境需求：建議使用 CUDA 11.8 以上版本，並安裝相關依賴套件：Bashpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install diffusers transformers pandas matplotlib accelerate

**硬體配置：**
硬體建議：GPU: 至少 8GB VRAM (本專案已啟用 attention\_slicing 以優化記憶體)。Disk: 預留約 10GB 空間供模型權重下載。🚀 使用說明1. 執行生成實驗執行 person\_b\_generator.py 將自動下載模型並開始 150 次（50 Prompts × 3 Exps）的生成任務。Bashpython code/person\_b\_generator.py

實驗過程中會每 10 張自動存檔一次 results.json 以防程式中斷。2. 數據分析與視覺化實驗完成後，執行分析腳本產出統計圖表：Bashpython code/analyze\_results.py

📊 監測指標Generation Time (s)：單張圖片推理耗時。Peak Memory Usage (GB)：利用 torch.cuda.max\_memory\_allocated() 監測 VRAM 峰值。Success Rate：記錄生成過程中的錯誤處理。提示：若需針對特定硬體進行優化（如使用 FP16 或不同的 Scheduler），可於 PersonB\_Generator 類別中修改 pipe 設定。
