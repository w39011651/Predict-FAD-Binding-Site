from sklearn.metrics import classification_report
import numpy as np
from transformers import Trainer

def get_classification_report(trainer:Trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions  # 模型輸出的 raw logits
    labels = predictions.label_ids    # 真實標籤
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    filtered_preds = preds[mask]
    filtered_labels = labels[mask]

    print(classification_report(filtered_labels.flatten(), filtered_preds.flatten()))

def custom_get_classification_report(trainer:Trainer, test_dataset, threshold=0.5):
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions  # 模型輸出的 raw logits
    labels = predictions.label_ids    # 真實標籤
    print(logits)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > threshold).astype(int)


    print(classification_report(labels.flatten(), preds.flatten()))