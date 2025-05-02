from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 將logits轉為二元預測（>0為正類）
    predictions = (logits[0] > 0).astype(np.int32)
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