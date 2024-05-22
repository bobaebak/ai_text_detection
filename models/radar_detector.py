import argparse 
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def detect(text, tokenizer=None, model=None):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")

    if not model:
        model = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")

    model.eval()
    model.to(device)

    encoded_inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512, 
        return_tensors="pt"
    )

    encoded_inputs = encoded_inputs.to(device)

    # outputs
    output = model(
        encoded_inputs.input_ids, 
        encoded_inputs.attention_mask, 
    )

    # probability
    logits = output.logits
    # prob = F.softmax(logits, dim=-1)[:, :].detach().cpu().numpy().squeeze()
    prob = F.log_softmax(logits, dim=-1)[:,0].exp().tolist()

    torch.mps.empty_cache() if device == "mps" else torch.cuda.empty_cache() if device == "cuda" else None
    
    return {"Fake": prob[0], "Real": 1-prob[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    model = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")

    prob = detect(args.text, tokenizer, model)
    print(prob)
