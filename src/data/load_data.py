import json
import pandas as pd
from models.transformers_model import ProteinDataset
from torch.utils.data import random_split

FILE_PATH = "../../data/raw/FAD_rmsim.json"

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

def sliding_window(protein_info, window_size = 13):
    ret = []
    left = 0
    right = window_size
    length = len(protein_info['sequence'])
    
    while right + 1 != length:#because slice is interval[left:right), so we can slice to idx, where idx is the length of sequence
        subseq_info = {"sequence":None, "label":None, "target_amino_acid":None}
        subseq_info['sequence'] = protein_info['sequence'][left:right]
        subseq_info['label'] = protein_info['label'][left:right]
        subseq_info['target_amino_acid'] = protein_info['sequence'][right-1]
        ret.append(subseq_info)
        left += 1
        right += 1
    
    return ret

def data_preprocess()->pd.DataFrame:
    with open(FILE_PATH) as f:
        data = json.load(f)

    protein_information = []

    for protein in data['results']:
        info = {"sequence_name":None, "sequence":None, "label":None}
        info['sequence_name'] = protein['primaryAccession']
        info['sequence'] = protein['sequence']['value']
        info['label'] = generate_label(protein, get_binding_site(protein['features']))
        protein_information.append(info)

    aa_info = []
    for protein in protein_information:
        aa_info += sliding_window(protein)

    df = pd.DataFrame(aa_info)

    return df

def split_dataset(df:pd.DataFrame):
    dataset = ProteinDataset(df)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return [train_dataset, test_dataset]

def get_dataframe():
    return data_preprocess()

def get_protein_dataset():
    """
    Return [train_dataset, test_dataset]
    """
    return split_dataset(data_preprocess())