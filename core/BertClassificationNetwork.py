import torch
import torch.nn as nn
from transformers import AutoModel
from DownstreamModel.CNNHead import CNNHead
from core.AttentionPooling import AttentionPooling
from core.BertBase import BertBasedNetwork
class BertClassificationNetwork(BertBasedNetwork):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    
    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss
    
    def single_batch_predict(self, batch):       
        input_ids = batch["input_ids"]          
        attention_mask = batch["attention_mask"]

        with torch.no_grad():                  
            output = self.forward(input_ids, attention_mask)   
            logits =  output["logits"]      
            probs = torch.softmax(logits, dim=-1)   
            preds = torch.argmax(probs, dim=-1) 
        return preds