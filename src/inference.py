import json
import torch
from config.path_config import THRESHOLD_FILE_PATH

def predict_call(text,model,tokenizer,device,MAX_LEN,LABELS):
    model.eval()

    # tokenize input text
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

    # # move tensors to device
    # input_ids = tokens["input_ids"].to(device)
    # attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        logits, score = model(tokens["input_ids"], tokens["attention_mask"])

    # convert logits to probabilities
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # load thresholds safely
    try:
        with open(THRESHOLD_FILE_PATH, "r") as f:
            THRESHOLDS = json.load(f)
            print("Thresholds found")
    except FileNotFoundError:
        # THRESHOLDS = {label: 0.5 for label in LABELS}
        THRESHOLDS = {
            "repetition": 0.35,
            "unclear_intro": 0.75,
            "missing_confirmation": 0.45,
            "long_agent_monologue": 0.65,
            "compliance_risk": 0.30
            }

        if "Agent_intro_present: True" in text:
            issues = [i for i in issues if i != "unclear_intro"]

        if "Confirmation_asked: True" in text:
            issues = [i for i in issues if i != "missing_confirmation"]

        if "long_agent_monologue" in issues:
            idx = LABELS.index("missing_confirmation")
            if probs[idx] > 0.35:
                issues.append("missing_confirmation")

        with open(THRESHOLD_FILE_PATH, "w") as f:
            json.dump(THRESHOLDS, f, indent=2)
        print("Threshold file created with default values")

    # apply thresholds
    issues = [
        LABELS[i]
        for i, p in enumerate(probs)
        if p > THRESHOLDS.get(LABELS[i], 0.5)
    ]

    return {
        "issues": issues,
        "quality_score": round(float(score.item() * 100), 2)
    }