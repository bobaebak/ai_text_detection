import argparse 
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def detect(text, tokenizer=None, model=None):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-large-openai-detector")

    if not model:
        model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-large-openai-detector")

    model = model.to(device)
    # encode
    encoded_inputs = tokenizer.encode_plus(
        text,   # Tokenize the sentence.
        None,   # Prepend the `[CLS]` token to the start.
        add_special_tokens=True,
        pad_to_max_length=True,
        max_length=512,
        return_token_type_ids=True,
        truncation=True,
        return_tensors="pt",
    )

    encoded_inputs = encoded_inputs.to(device)

    # outputs
    output = model(
        encoded_inputs.input_ids, 
        encoded_inputs.attention_mask, 
        encoded_inputs.token_type_ids,
    )

    # probability
    logits = output.logits
    prob = F.softmax(logits, dim=-1)[:, :].detach().cpu().numpy().squeeze()
    
    torch.mps.empty_cache() if device == "mps" else torch.cuda.empty_cache() if device == "cuda" else None

    return {"Fake": prob[0], "Real": prob[1]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-large-openai-detector")
    model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-large-openai-detector")

    prob = detect(args.text, tokenizer, model)
    print(prob)
