import torch
import torch.nn as nn
from transformers import AutoModel
from core.AttentionPooling import AttentionPooling

class BertBasedNetwork(nn.Module):
    def __init__(self, downstreamNN: nn.Module, backbone_name: str = "microsoft/deberta-v3-base", hidden_size=768, pooling = "average"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.downstreamNN = downstreamNN
        self.hidden_size = hidden_size
        self.attention_pooling = AttentionPooling(hidden_size)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask,labels = None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        pooling = self.pooling
        hidden_states = None
        if pooling == "average":
            # average pooling  
            masked = last_hidden * attention_mask.unsqueeze(-1)
            hidden_states = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif pooling == "max":
            #  max pooling
            masked = last_hidden.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            hidden_states, _ = masked.max(dim=1)

        elif pooling == "attention":
            # attention pooling
            hidden_states = self.attention_pooling(last_hidden, attention_mask)

        else:  # CLS
            hidden_states = last_hidden[:, 0, :]

        if hidden_states is None:
            raise ValueError("No hidden states computed!")

        logits = self.downstreamNN(hidden_states)
        ## Compatible to Trainer
        loss = None
        if labels is not None :
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
    
    def single_batch_predict(self, batch):
        input_ids = batch["input_ids"]          
        attention_mask = batch["attention_mask"]

        with torch.no_grad():                  
            output = self.forward(input_ids, attention_mask)   
            logits =  output["logits"]      
            probs = torch.softmax(logits, dim=-1)   
            preds = torch.argmax(probs, dim=-1) 
        return preds




