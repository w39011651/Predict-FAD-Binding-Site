from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from torchmetrics import Precision, Recall, F1Score, MatthewsCorrCoef
import numpy as np
import torch

def sklearn_compute_metrics(eval_pred, threshold = 0.5):
    logits, labels = eval_pred
    print(logits)
    print(labels)
    logits = np.array(logits)
    labels = np.array(labels)
    # 應用遮罩，與CustomTrainer一致
    valid_mask = (labels != -100)
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    # 將logits轉為二元預測（>0為正類）
    predictions = (logits[0] > threshold).astype(np.int32)
    # 確保標籤為整數
    labels = labels.astype(np.int32)
    # 計算指標
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    mcc = matthews_corrcoef(labels.flatten(), predictions.flatten())
    return {
        "precision_0": precision[0],  # 負類precision
        "recall_0": recall[0],        # 負類recall（特異度）
        "f1_0": f1[0],                # 負類F1
        "precision_1": precision[1],  # 正類precision
        "recall_1": recall[1],        # 正類recall
        "f1_1": f1[1],                # 正類F1
        "mcc": mcc,                   # Matthews相關係數
        "specificity": recall[0]      # 特異度（與recall_0相同）
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
precision_metric = Precision(task='multiclass', average='none', num_classes=2).to(device)
recall_metric = Recall(task='multiclass', average='none', num_classes=2).to(device)
f1_metric = F1Score(task='multiclass', average='none', num_classes=2).to(device)
mcc_metric = MatthewsCorrCoef(task='binary').to(device)


def torch_compute_metrics(eval_pred, threshold = 0.5):
    model_output, labels = eval_pred

    #Be sure that labels is np array
    logits_np = model_output[0]
    labels_np = np.asarray(labels)
    
    logits_tensor = torch.from_numpy(logits_np).to(device)
    labels_tensor = torch.from_numpy(labels_np).to(device).long()

    # 應用遮罩，與CustomTrainer一致
    valid_mask = (labels_tensor != -100)
    filtered_logits = logits_tensor[valid_mask]
    filtered_labels = labels_tensor[valid_mask]

    if filtered_labels.ndim == 1:
        predictions = (filtered_logits > 0).long()# 將logits轉為二元預測（>0為正類）
    elif filtered_labels.ndim == 2:
        print(f"Warning: Logits shape after masking is {filtered_logits.shape}. Assuming independent binary classification per element.")
        # 展平處理 - 這可能不是你想要的，取決於你的標籤如何定義！
        # 如果你的 labels 也是 (M, 512)，並且你想比較每個位置：
        # predictions = (filtered_logits > threshold).long() # 保持二維
        # 如果你的 labels 是 (M,)，代表序列級別，那麼 logits 如何對應？需要修改模型或邏輯
        # --- 暫時以展平處理，你需要根據任務調整 ---
        predictions = (filtered_logits.view(-1) > threshold).long()
        filtered_labels = filtered_labels.view(-1) # 同樣展平標籤以匹配

     # --- 指標計算 ---
    prec = precision_metric(predictions, filtered_labels)
    rec = recall_metric(predictions, filtered_labels)
    f1 = f1_metric(predictions, filtered_labels)
    mcc = mcc_metric(predictions, filtered_labels)

    metrics = {
        "precision_0": prec[0].item(), "recall_0": rec[0].item(), "f1_0": f1[0].item(),
        "precision_1": prec[1].item(), "recall_1": rec[1].item(), "f1_1": f1[1].item(),
        "mcc": mcc.item(), "specificity": rec[0].item()
    }
    return metrics