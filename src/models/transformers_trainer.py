import os
import pickle
import tqdm
import torch
import torch.nn
from transformers import EsmTokenizer, EsmModel
from transformers import EsmForTokenClassification, Trainer, TrainingArguments
from src.data import load_data
from src.utils.helper import log


tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
model = EsmForTokenClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=2)
df = load_data.data_preprocess()

def prepare_inputs(row):
    encoding = tokenizer(row["sequence"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = encoding['input_ids']
    # 將 labels 補齊或截斷成一樣長度
    label = row["label"]  # -100 是 PyTorch ignore_index

    max_len = encoding['input_ids'].shape[1]
    if len(label) < max_len:
        label += [-100] * (max_len - len(label))
    elif len(label) > max_len:
        label = label[:max_len]

    log("Process row:{}...".format(row["sequence"]))
    
    
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(label)
    }

def make_pickel_file():
    if os.path.isfile("processed_dataset.pkl") is False:
        processed_samples = []
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            processed = prepare_inputs(row)
            processed_samples.append(processed)

        with open("processed_dataset.pkl", "wb") as f:
            pickle.dump(processed_samples, f)

def training(train_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        #evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # 真實情況下應分出驗證集
    )
    trainer.train()

def run():
    make_pickel_file()
    train_dataset, test_dataset = load_data.get_protein_dataset()
    training(train_dataset, test_dataset)
