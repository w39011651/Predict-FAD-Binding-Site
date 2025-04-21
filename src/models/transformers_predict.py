from transformers import EsmForTokenClassification


model = EsmForTokenClassification.from_pretrained('facebook/esm2_t6_8M_UR50D')
