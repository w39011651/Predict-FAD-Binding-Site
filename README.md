# 預測FAD/FMN Binding-Site

## Introduction

這個專題主要是透過各種機器學習來預測FAD/FMN Binding Site
目前完成: Transformer(Fine-tune model facebook/esm2_t6_8M_UR50D)

## Notice
Fine-tune的模型過大，無法推送至github，第一次運行需要在main.py中使用
```
transformers_trainer.run()
```
以得到模型

參考時間:

GPU: NVIDIA GeForce RTX 4060 Laptop GPU

時間: About 40 minutes

## 目錄結構(From ChatGPT):
```
protein_ml_project/
├── config/
│   └── config.yaml             # 參數設定（模型、訓練、資料等）
├── data/
│   ├── raw/                    # 原始蛋白質序列或FASTA檔案
│   ├── processed/              # 處理後的資料集（CSV、Numpy、Pickle等）
│   └── features/               # 特徵資料（embedding、AAC、PSSM等）
├── notebooks/
│   └── EDA.ipynb               # 初步探索性分析
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py        # 蛋白質資料載入與預處理
│   │   └── feature_engineering.py  # 特徵萃取，如 one-hot、ProtBERT 等
│   ├── models/
│   │   ├── rf_model.py         # Random Forest 模型
│   │   ├── cnn_model.py        # CNN 模型
│   │   ├── transformer.py      # Transformer 模型（預留 Mamba 架構）
│   │   └── train.py            # 訓練通用邏輯（支援不同模型）
│   ├── evaluation/
│   │   └── metrics.py          # 評估指標、混淆矩陣等
│   └── utils/
│       └── helper.py           # 通用工具（logging、seed 設定等）
├── experiments/
│   └── run_2025_04_15_rf.yaml  # 實驗紀錄
├── outputs/
│   ├── models/                 # 存放訓練好的模型檔
│   ├── logs/                   # 訓練過程記錄
│   └── results/                # 結果輸出（圖表、CSV等）
├── requirements.txt
├── README.md
└── main.py                    # 可選的主入口腳本（支援 argparse 指定模型）
```

💡 架構設計思路與優點
🔁 模型模組化
每種模型（RF, CNN, Transformer, Mamba）都在 models/ 下作為獨立檔案

訓練邏輯統一由 train.py 控制，可插入不同模型

📊 資料分層
raw 資料 ↔ 處理後資料 ↔ 特徵資料 清楚分開

資料處理與特徵工程為獨立模組，便於重複使用

⚙️ 可設定化
所有訓練超參與路徑存在 config.yaml，方便實驗比較

🧪 實驗紀錄
每次訓練參數、模型結果用 YAML 或 JSON 存放在 experiments/

🔄 未來擴充容易
你可以很輕鬆地加入 Mamba 模型，只需在 models/mamba.py 裡定義它

## Results

### Transformer:

```
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     88813
           1       0.93      0.88      0.90      2889

    accuracy                           0.99     91702
   macro avg       0.96      0.94      0.95     91702
weighted avg       0.99      0.99      0.99     91702
```

## Reference
[Quang-Thai Ho, Trinh-Trung-Duong Nguyen, Nguyen Quoc Khanh Le, Yu-Yen Ou,
FAD-BERT: Improved prediction of FAD binding sites using pre-training of deep bidirectional transformers,
Computers in Biology and Medicine,
Volume 131,
2021,
104258,
ISSN 0010-4825,
https://doi.org/10.1016/j.compbiomed.2021.104258.](https://www.sciencedirect.com/science/article/abs/pii/S0010482521000524?casa_token=kEkRv3ranIUAAAAA:S5dHeB0riirLnAALU0PxfWk71ubyxDTLdXzXXxN45KTIWii8kmXzVG8uSOmnfr0UB8NCs00Mhg#sec10)

[Peng, F.Z., Wang, C., Chen, T. et al. PTM-Mamba: a PTM-aware protein language model with bidirectional gated Mamba blocks. Nat Methods (2025).](https://doi.org/10.1038/s41592-025-02656-9)

[Albert Gu, Tri Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
