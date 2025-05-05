import yaml
import torch
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from src.models.ProteinCLassifier import ProteinClassifier, CustomTrainer
from src.utils.compute_metrics import sklearn_compute_metrics, torch_compute_metrics

model = ProteinClassifier('bert-base-multilingual-cased')

def training(train_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open("../config/bert_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 轉成 TrainingArguments 物件
    training_args = TrainingArguments(**config['training_arguments'])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # 真實情況下應分出驗證集
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()


def customTraining(train_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open("config/bert_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 轉成 TrainingArguments 物件
    training_args = TrainingArguments(**config['training_arguments'])

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # 真實情況下應分出驗證集
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        #compute_metrics= sklearn_compute_metrics
    )

    trainer.train()
    torch.cuda.empty_cache()
    trainer.save_model(training_args.output_dir)
    return trainer