import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from config.path_config import (
    load_config,
    ALL_DATA_FILE_PATH,
    TRAIN_DATA_FILE_PATH,
    TEST_DATA_FILE_PATH,
    VAL_DATA_FILE_PATH,
    SAMPLE_DATA_FILE_PATH,
    MODEL_PATH
)

from src.data_ingestion import DataIngestion, conversation_to_structured_text
from src.train import (
    LABELS,
    CallDataset,
    Call_Model,
    train_one_epoch,
    validate,
    compute_class_weights
)
from src.inference import predict_call


# -------------------- SETUP --------------------
def setup():
    raw_config = load_config()

    config = {
        "MODEL_NAME": raw_config["MODEL_NAME"],
        "BATCH_SIZE": int(raw_config["BATCH_SIZE"]),
        "MAX_LEN": int(raw_config["MAX_LEN"]),
        "LR": float(raw_config["LR"]),
        "EPOCHS": int(raw_config["EPOCHS"]),
    }

    tokenizer = DistilBertTokenizerFast.from_pretrained(config["MODEL_NAME"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return config, tokenizer, device


# -------------------- DATA --------------------
def prepare_data():
    raw_data = DataIngestion._read_train_data_file(ALL_DATA_FILE_PATH)
    sample_df = DataIngestion.extract_sample_data(raw_data, SAMPLE_DATA_FILE_PATH)

    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    sample_df["flattened_text"] = sample_df["conversation"].apply(
        conversation_to_structured_text
    )

    train_df = sample_df.iloc[:1500]
    test_df  = sample_df.iloc[1500:1800]
    val_df   = sample_df.iloc[1800:2000]

    train_df.to_json(TRAIN_DATA_FILE_PATH, orient="records", indent=2)
    test_df.to_json(TEST_DATA_FILE_PATH, orient="records", indent=2)
    val_df.to_json(VAL_DATA_FILE_PATH, orient="records", indent=2)

    return train_df, val_df, test_df


def build_label_tensor(df):
    labels = []
    for _, row in df.iterrows():
        issues = row.get("call_analysis", {}).get("issues", [])
        labels.append([1.0 if label in issues else 0.0 for label in LABELS])

    return torch.tensor(labels, dtype=torch.float32)


# -------------------- TRAINING --------------------
def train_model(train_df, val_df, tokenizer, device, config):
    train_labels = build_label_tensor(train_df)

    train_loader = DataLoader(
        CallDataset(train_df, tokenizer, config["MAX_LEN"]),
        batch_size=config["BATCH_SIZE"],
        shuffle=True
    )

    val_loader = DataLoader(
        CallDataset(val_df, tokenizer, config["MAX_LEN"]),
        batch_size=config["BATCH_SIZE"],
        shuffle=False
    )

    model = Call_Model().to(device)

    # Freeze BERT initially
    for p in model.bert.parameters():
        p.requires_grad = False

    # Load saved model if exists
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        model.eval()
        return model

    optimizer = AdamW(model.parameters(), lr=config["LR"])
    pos_weights = compute_class_weights(train_labels, len(LABELS)).to(device)

    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    reg_loss = nn.MSELoss()

    for epoch in range(config["EPOCHS"]):
        print(f"\nEpoch {epoch + 1}/{config['EPOCHS']}")

        train_one_epoch(model, train_loader, optimizer, device, cls_loss, reg_loss)
        validate(model, val_loader, device, cls_loss, reg_loss)

        # Unfreeze BERT after 3rd epoch
        if epoch == 2:
            print("Unfreezing BERT layers...")
            for p in model.bert.parameters():
                p.requires_grad = True

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

    return model


# -------------------- INFERENCE --------------------
def run_inference(model, tokenizer, device, config, conversation):
    model.eval()

    text = conversation_to_structured_text(conversation)
    prediction = predict_call(
        text,
        model,
        tokenizer,
        device,
        config["MAX_LEN"],
        LABELS
    )

    print("\nTEXT:\n", text)
    print("\nPREDICTION:\n", prediction)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    config, tokenizer, device = setup()

    train_df, val_df, test_df = prepare_data()

    model = train_model(train_df, val_df, tokenizer, device, config)

    conversation = [
        {
            "speaker": "customer",
            "text": "Hello, I'd like to request a refund for a product I purchased about two months ago."
        },
        {
            "speaker": "agent",
            "text": "Thank you for contacting ShopEase customer support. My name is Riya."
        }
    ]

    run_inference(model, tokenizer, device, config, conversation)
