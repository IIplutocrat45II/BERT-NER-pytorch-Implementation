#Importing the dependencies
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

# Network model : Net
class Net(nn.Module):

    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False):

        super().__init__()

        # We use the BERT base cased model for the NER task
        self.bert = BertModel.from_pretrained('bert-base-cased')

        # We initialize a LSTM decoder --> Feature based approach
        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)

        # Output layer
        self.fc = nn.Linear(768, vocab_size)

        self.device = device
        self.finetuning = finetuning


    def forward(self, x, y, ):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training and self.finetuning:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
