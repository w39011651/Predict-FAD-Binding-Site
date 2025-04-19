from src.models.transformers_predict import model
from src.data.load_data import get_protein_dataset
from sklearn.metrics import classification_report
from transformers import Trainer
import numpy as np

def get_classification_report(test_dataset):
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions  # 模型輸出的 raw logits
    labels = predictions.label_ids    # 真實標籤
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    filtered_preds = preds[mask]
    filtered_labels = labels[mask]

    print(classification_report(filtered_labels.flatten(), filtered_preds.flatten()))

def run():
    train_dataset, test_dataset = get_protein_dataset()
    get_classification_report(test_dataset)
