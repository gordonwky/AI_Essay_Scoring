import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict

class EssayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        max_len: int = 512,
        stride: int = 256
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.stride = stride

        self.encodings = []
        self.labels = []
        self.essay_ids = []

        # 每篇 essay 可能切成多個 chunk
        df["token"] = df["full_text"].apply(
            lambda x: self.tokenizer(
                x,
                max_length=self.max_len,
                stride=self.stride,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="pt"
            )
        )

        for _, row in df.iterrows():
            tokenized = row["token"]
            num_chunk = tokenized["input_ids"].shape[0]

            for i in range(num_chunk):
                self.encodings.append({
                    "input_ids": tokenized["input_ids"][i],
                    "attention_mask": tokenized["attention_mask"][i]
                })
                # ✅ 1–6 分數 → 0–5 class index
                self.labels.append(int(row["score"]) - 1)
                self.essay_ids.append(int(row["essay_id"]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            key: val.clone().detach() for key, val in self.encodings[idx].items()
        }
        # ✅ CrossEntropyLoss 要 long tensor
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["essay_id"] = torch.tensor(self.essay_ids[idx], dtype=torch.long)
        return item
