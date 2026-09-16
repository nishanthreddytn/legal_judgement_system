import torch
import torch.nn as nn
from TorchCRF import CRF
from transformers import AutoModel

class HybridLegalNER(nn.Module):
    def __init__(self, base_model, num_labels, hidden_size=256, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.id2label = {}

    def emissions(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.classifier(self.dropout(x))

    def forward(self, input_ids, attention_mask, labels=None):
        e = self.emissions(input_ids, attention_mask)
        if labels is None:
            return self.crf.decode(e, mask=attention_mask.bool())
        y = labels.clone()
        y[y == -100] = 0
        return -self.crf(e, y, mask=attention_mask.bool()).mean()

    def predict(self, input_ids, attention_mask):
        return self.crf.decode(self.emissions(input_ids, attention_mask), mask=attention_mask.bool())
