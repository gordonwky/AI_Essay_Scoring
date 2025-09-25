import argparse
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from core.EssayDataset import EssayDataset 
from core.BertClassificationNetwork import BertClassificationNetwork
from core.BertOrdinalNetwork import BertOrdinalNetwork
from DownstreamModel.MLPHead import MLPHead
from DownstreamModel.CNNHead import CNNHead
from core.utils import compute_metrics, freeze_model


# ===== prepare dataset =====
def prepare_datasets(df, sample_frac=None):
    if sample_frac is None:   
        selected_train_df = df.copy()
    else:
        rare_6 = df[df["score"] == 6]

        rare_1_5 = df[df["score"].isin([1, 5])]

        # rare_6 = df[df["score"] == 6].sample(frac=sample_frac, random_state=42)

        # rare_1_5 = df[df["score"].isin([1, 5])].groupby("score", group_keys=False).apply(
        #     lambda x: x.sample(frac=sample_frac, random_state=42)
        # )
        common = df[df["score"].isin([2, 3, 4])].groupby("score", group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=42)
        )
        selected_train_df = pd.concat([rare_6, rare_1_5, common]).reset_index(drop=True)

    selected_train_df["essay_id"] = pd.factorize(selected_train_df["essay_id"])[0] + 1
    df_train, df_val = train_test_split(
        selected_train_df,
        test_size=0.2,
        random_state=42,
        stratify=selected_train_df["score"],
    )
    return EssayDataset(df_train), EssayDataset(df_val)


# ===== main =====
def main(args):
    model_name = "microsoft/deberta-v3-base"
    output_dim = 6  # scores from  to 6
    # load data
    train_df = pd.read_csv("./learning-agency-lab-automated-essay-scoring-2/train.csv")
    train_dataset, val_dataset = prepare_datasets(train_df, args.sample_frac)

    # model
    def downstream_init(output_dim):
        if args.downstream == "cnn":    
            downstream = CNNHead(output_dim= output_dim)
        elif args.downstream == "mlp":
            downstream = MLPHead(output_dim=output_dim)
        return downstream
    # ordinal or classification
    if args.task == "ordinal":
        output_dim = output_dim - 1  # for ordinal, num_classes = num_labels - 1
        downstream = downstream_init(output_dim)
        model = BertOrdinalNetwork(num_classes=6, downstreamNN=downstream, backbone_name=model_name)
    else:
        downstream = downstream_init(output_dim)
        model = BertClassificationNetwork(num_classes=6, downstreamNN=downstream, backbone_name=model_name)

    # freeze layers
    if args.freeze > 0:
        freeze_model(model, n_layers=args.freeze)

    training_args = TrainingArguments(
        output_dir="./essay_cls",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        logging_dir="./logs",
        report_to="none",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./essay_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=int, default=0, help="freeze last N layers of backbone (0 = no freeze)")
    parser.add_argument("--sample_frac", type=float, default=None, help="Fraction of data to sample (None = use full dataset)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--downstream", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "ordinal"])
    args = parser.parse_args()

    main(args)
