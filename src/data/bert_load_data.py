import json
import torch

def read_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    return data

def sliding_window(protein_info, window_size = 13):
    ret = []
    left = 0
    right = window_size
    length = len(protein_info['sequence'])
    
    while right + 1 != length:#because slice is interval[left:right), so we can slice to idx, where idx is the length of sequence
        subseq_info = {"sequence":None, "label":None, "target_amino_acid":None}
        subseq_info['sequence'] = protein_info['sequence'][left:right]
        subseq_info['label'] = protein_info['label'][left:right]
        subseq_info['target_amino_acid'] = protein_info['sequence'][(left+right)//2]
        ret.append(subseq_info)
        left += 1
        right += 1
    
    return ret

def get_binding_site(features):
    binding_site = []
    for feature in features:
        if feature['type'] == 'Binding site' and (feature['ligand']['name']=='FAD' or feature['ligand']['name']=='FMN'):#only need FAD
            binding_site.append([feature['location']['start']['value'], feature['location']['end']['value']])
    return binding_site

def generate_label(protein, positions):
    n = protein['sequence']['length']
    labels = [0 for _ in range(n)]

    for bind_site in positions:
        for i in range(bind_site[0], bind_site[1]+1):
            labels[i-1] = 1
    return labels

def get_protein_information(json_data):
    protein_information = []

    for protein in json_data['results']:
        info = {"sequence_name":None, "sequence":None, "label":None}
        info['sequence_name'] = protein['primaryAccession']
        info['sequence'] = protein['sequence']['value']
        info['label'] = generate_label(protein, get_binding_site(protein['features']))
        protein_information.append(info)

    return protein_information

def get_aa_info(protein_information):
    aa_information = []
    for protein in protein_information:
        aa_information += sliding_window(protein) 
    return aa_information

MAX_LEN = 512

def get_pretoken(amino_acid_information):
    ret = []
    for aa_info in amino_acid_information:

        while len(aa_info['sequence']) > MAX_LEN:##limit the size of each sequence in MAX_LEN
            nlp_pretoken = list(aa_info['sequence'][:MAX_LEN])
            label_pretoken = aa_info['label'][:MAX_LEN]
            ret.append([nlp_pretoken, label_pretoken])
            aa_info['sequence'] = aa_info['sequence'][MAX_LEN:]
            aa_info['label'] = aa_info['label'][MAX_LEN:]

        nlp_pretoken = list(aa_info['sequence'])
        label_pretoken = aa_info['label']
        ret.append([nlp_pretoken, label_pretoken])

    return ret

def tokenize(tokenizer, nlp_pretoken, label_pretoken):
    encoding = tokenizer(nlp_pretoken, 
                         is_split_into_words=True, 
                         add_special_tokens = True,
                         return_tensors="pt", 
                         padding="max_length", 
                         truncation=True, 
                         max_length=512
                         )
    
    word_ids = encoding.word_ids()
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:   
            label_ids.append(label_pretoken[word_idx])
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(label_ids, dtype=float),
    }

def customTokenize(tokenizer, nlp_pretoken, label_pretoken):
    # 將蛋白質序列轉為空格分隔格式（ProtBERT 需要）
    nlp_pretoken = " ".join(nlp_pretoken)
    encoding = tokenizer(
        nlp_pretoken,
        is_split_into_words=False,  # 已經手動分隔
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # 創建標籤，長度與 input_ids 一致
    label_ids = torch.zeros(512, dtype=torch.float32)  # 初始化為 0.0
    # 將原始標籤映射到 token 位置（跳過 [CLS] 和 [SEP]）
    for i, idx in enumerate(range(len(label_pretoken))):
        if i < 510:  # 留空間給 [CLS] 和 [SEP]
            label_ids[i + 1] = float(label_pretoken[idx])  # 從索引 1 開始

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": label_ids,
    }

from torch.utils.data import random_split
from src.models.protein_dataset import ProteinDataset

def split_dataset(dataset:ProteinDataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return [train_dataset, test_dataset]

from tqdm import tqdm
from transformers import BertTokenizerFast


def run():
    """RETURN [TRAIN_DATASET, TEST_DATASET]"""

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    data = read_from_json('data/raw/FAD_rmsim.json')

    len(data['results'])

    protein_information = get_protein_information(json_data=data)

    pretoken = get_pretoken(protein_information)

    token = []
    for pt in tqdm(pretoken):
        after_token = customTokenize(tokenizer, pt[0], pt[1])
        print(f"Size of input_ids:{len(after_token["input_ids"])},label:{len(after_token["labels"])}")
        token.append(after_token)

    [train_dataset, test_dataset] = split_dataset(token)
    return [train_dataset, test_dataset]
