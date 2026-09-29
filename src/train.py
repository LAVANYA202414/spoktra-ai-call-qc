import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import logging
from torch.utils.data import Dataset
from config.path_config import load_config
from config.path_config import (MODEL_PATH)
from transformers import DistilBertModel, DistilBertTokenizerFast



# to remove login errors:
logging.set_verbosity_error()
# to remove progress bars:
logging.disable_progress_bar()


# config:
config = load_config()
MODEL_NAME = config["MODEL_NAME"]
MAX_LEN = int(config["MAX_LEN"])   
LR = float(config.get("LR", 5e-5))
LABELS = [
    "repetition",
    "unclear_intro",
    "missing_confirmation",
    "long_agent_monologue",
    "compliance_risk"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# ==> CREATE PYTORCH DATASET
class CallDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.df)

        # Converts text → token IDs , Truncates long calls , Pads short calls
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tokens = self.tokenizer(
            row["flattened_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        issues = row["call_analysis"]["issues"]
        label_vector = [
            1.0 if label in issues else 0.0
            for label in LABELS
        ]
        quality_score = row["call_analysis"]["quality_score"]
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_vector, dtype=torch.float),
            "score": torch.tensor(quality_score, dtype=torch.float),
        }



# custom neural network
class Call_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(768, len(LABELS))
        self.regressor = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        # CLS token -> for classification tasks
        x = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        labels = self.classifier(x)
        score = torch.sigmoid(self.regressor(x)).squeeze(-1)
        return labels, score


# model = Call_Model().to(device)

# # compute class weights from training data:
# def compute_class_weights(df,labels):
#     counts = torch.zeros(len(labels))
#     for issues in df["call_analysis"].apply(lambda x:x["issues"]):
#         for i , label in enumerate(labels):
#             if label in issues:
#                 counts[i] += 1

#     total = len(df)
#     pos_weight = (total - counts) / (counts + 1e - 6)

#     return pos_weight

def compute_class_weights(labels_tensor, num_classes):
    pos_counts = torch.zeros(num_classes)
    neg_counts = torch.zeros(num_classes)

    for y in labels_tensor:
        pos_counts += y
        neg_counts += (1 - y)

    pos_weights = torch.zeros(num_classes)

    for i in range(num_classes):
        if pos_counts[i] == 0:
            pos_weights[i] = 1.0
        else:
            pos_weights[i] = neg_counts[i] / pos_counts[i]

    # Clamp extreme values
    pos_weights = pos_weights.clamp(1.0, 20.0)

    # Reduce unclear_intro dominance
    unclear_idx = LABELS.index("unclear_intro")
    pos_weights[unclear_idx] *= 0.5

    return pos_weights



# for classification 
cls_loss_fn = nn.BCEWithLogitsLoss()
# for regression
reg_loss_fn = nn.MSELoss()
# optimizer:
# optimizer = AdamW(model.parameters(), lr=LR)


def train_one_epoch(model, data_loader, optimizer, device, cls_loss_fn, reg_loss_fn):
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets_cls = batch["labels"].to(device)   # shape: [B, num_labels]
        targets_reg = batch["score"].to(device)    # shape: [B]

        optimizer.zero_grad()

        out_cls, out_reg = model(ids, mask)

        # ---- classification loss (multi-label) ----
        loss_cls = cls_loss_fn(out_cls, targets_cls)
        loss_cls = loss_cls.mean()   # ⭐ IMPORTANT

        # ---- regression loss (quality score) ----
        loss_reg = reg_loss_fn(out_reg.squeeze(), targets_reg)
        loss_reg = loss_reg.mean()   # ⭐ IMPORTANT

        # ---- combined loss ----
        loss = loss_cls + 0.2 * loss_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(model, val_loader, device, cls_loss_fn, reg_loss_fn):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            scores = batch["score"].to(device)

            labels_pred, score_pred = model(input_ids, attention_mask)

            loss_cls = cls_loss_fn(labels_pred, labels)
            loss_reg = reg_loss_fn(score_pred, scores)

            val_loss += (loss_cls + loss_reg).item()

    return val_loss / len(val_loader)



    # cls_probs = torch.sigmoid(cls_logits)
    # all_probs.append(cls_probs.cpu())
    # all_targets.append(cls_labels.cpu)