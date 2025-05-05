from torch import nn
from torch import stack, relu
from transformers import AutoModel, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import tensor
from torchvision.ops import sigmoid_focal_loss as focol_loss


class ProteinClassifier(nn.Module):
    def __init__(self, pretrained_model_name, dropout=0.3):
        super(ProteinClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # 1D CNN
        self.conv1 = nn.Conv1d(self.bert.config.hidden_size, 64, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        # 二分類輸出層：hidden_size -> 1（每個token輸出一個概率）
        #self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.classifier = nn.Linear(32, 1)
        # init the weight
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

        #self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=tensor([55.0]))
        self.loss_fn = focol_loss

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True)
        hidden_state = outputs.hidden_states
        last_four_layers = hidden_state[-4:]
        sequence_output = stack(last_four_layers, dim=0).sum(dim=0)# (batch_size, seq_len, hidden_size) sum up last 4 layers
        sequence_output = sequence_output.permute(0, 2, 1)
        #sequence_output = self.dropout(sequence_output)
        x = relu(self.conv1(sequence_output))
        x = relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        logits = self.classifier(x).squeeze(-1)  # (batch_size, seq_len)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return SequenceClassifierOutput(
            loss = loss,
            logits = logits,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions
        )
    
class CustomTrainer(Trainer):
    def compute_loss(self, model:ProteinClassifier, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # 獲取標籤
        outputs = model(**inputs)  # 前向傳播
        logits = outputs.logits
        loss_fn = model.loss_fn.to(logits.device)

        valid_mask = (inputs["attention_mask"] == 1) & (labels != -100)
        loss = loss_fn(logits[valid_mask], labels[valid_mask].float())
        #loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
