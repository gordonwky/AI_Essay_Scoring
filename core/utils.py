from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np
import torch.nn as nn
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)      # 0..5
    acc = accuracy_score(labels, preds)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    return {"accuracy": acc, "qwk": qwk}
def freeze_model(model, n_layers=0):
    """
    Freeze the first n_layers of the model.
    Args:
        model: The model to be frozen.
        n_layers: The number of layers to be frozen.
    """
    # Freeze embeddings
    for param in model.backbone.embeddings.parameters():
        param.requires_grad = False

    # Freeze encoder layers
    if n_layers > 0:
        for layer in model.backbone.encoder.layer[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    # Ensure the rest of the model is trainable
    for p in model.downstreamNN.parameters():
        p.requires_grad = True
