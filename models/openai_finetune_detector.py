import os
import json
import argparse 
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURR_DIR)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def detect(text, tokenizer=None, model=None):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")

    if not model:
        model = torch.load("".join(PARENT_DIR+"/models/"+args.model_name))
    
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
    return {"Fake": prob[0], "Real": prob[1]}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--model_name", type=str, default="fine_tune_epoch1.pth")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
    model = torch.load("".join(PARENT_DIR+"/models/"+args.model_name))

    prob = detect(args.text, tokenizer, model)
    print(prob)

