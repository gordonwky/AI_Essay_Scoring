import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from core.EssayDataset import EssayDataset  # 你自己寫的 Dataset class
from core.BertBase import BertBasedNetwork
from DownstreamModel.MLPHead import MLPHead
from core.utils import compute_metrics, freeze_model
def prepare_datasets(df):
    # sampling for prediction 
    # Include all rare scores
    # Score 6: Keep all
    rare_6 = df[df["score"] == 6].apply(
        lambda x: x.sample(frac=0.5, random_state=42)
    )

    # Score 1 & 5: Keep 50%
    rare_1_5 = df[df["score"].isin([1, 5])].groupby("score", group_keys=False).apply(
        lambda x: x.sample(frac=0.05, random_state=42)
    )

    # Scores 2, 3, 4: Sample 15%
    common = df[df["score"].isin([2, 3, 4])].groupby("score", group_keys=False).apply(
        lambda x: x.sample(frac=0.02, random_state=42)
    )
    # Combine all sampled subsets
    selected_train_df = pd.concat([rare_6, rare_1_5, common]).reset_index(drop=True)
    selected_train_df["essay_id"] = pd.factorize(selected_train_df["essay_id"])[0] + 1
    df_train, df_val = train_test_split(
        selected_train_df,
        test_size=0.2,         
        random_state=42,       
        stratify=selected_train_df["score"])

    train_dataset = EssayDataset(df_train)
    val_dataset = EssayDataset(df_val)

    return train_dataset, val_dataset
def main():
    model_name = "microsoft/deberta-v3-base"

    # 修改成你嘅檔案路徑
    train_df = pd.read_csv("./learning-agency-lab-automated-essay-scoring-2/train.csv")
    print(train_df.head())
    train_dataset, val_dataset = prepare_datasets(train_df)

    downstream = MLPHead(output_dim = 6)
    model = BertBasedNetwork(downstream)

    training_args = TrainingArguments(
        output_dir="./essay_cls",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        logging_dir="./logs",
        report_to="none",   # 避免用 wandb
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 儲存模型
    trainer.save_model("./essay_model")

if __name__ == "__main__":
    main()
