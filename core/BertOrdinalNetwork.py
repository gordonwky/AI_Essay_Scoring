from core.BertBase import BertBasedNetwork
import torch
import torch.nn as nn

class BertOrdinalNetwork(BertBasedNetwork):
    def __init__(self,num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes - 1

    def compute_loss(self, logits, labels):
        # Assuming logits are of shape (batch_size, num_classes)
        # and labels are of shape (batch_size,) with class indices
        batch_size = logits.size(0)
        targets = torch.zeros((batch_size, self.num_classes), device=logits.device)
        for i in range(batch_size):
            targets[i, :labels[i]] = 1
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, targets)
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