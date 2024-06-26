import argparse 
import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def calc_perplexity(tokenizer, model, device, text):    
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    # likelihoods = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss * trg_len
            # neg_log_likelihood = outputs.loss
            # likelihoods.append(neg_log_likelihood)

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # ppl = torch.exp(torch.stack(nlls).mean())
    ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
    return ppl, nlls

def labeling(perplexity_per_line_avg) -> tuple:
    if perplexity_per_line_avg < 47:
        return "The Text is generated by AI.", 0, (47, 80)
    elif perplexity_per_line_avg < 80:
        return "The Text is most probably contain parts which are generated by AI. (require more text for better Judgement)", 0, (47, 80)
    else:
        return "The Text is written by Human.", 1, (47, 80)


def detect(text, tokenizer=None, model=None) -> dict:   
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side='left')
    if model is None:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    total_valid_char = re.findall("[a-zA-Z0-9]+", text)
    total_valid_char = sum([len(x) for x in total_valid_char]) # finds len of all the valid characters a sentence

    lines = re.split(r'(?<=[.?!][ \[\(])|(?<=\n)\s*', text)
    lines = list(filter(lambda x: (x is not None) and (len(x) > 0), lines))
    
    model = model.to(device)

    # perplexity of phrase
    perplexity, nlls = calc_perplexity(tokenizer, model, device, text)

    # perplexity on each line
    offset = ""
    perplexity_per_line = []
    for i, line in enumerate(lines):
        if re.search("[a-zA-Z0-9]+", line) == None:
            continue
        if len(offset) > 0:
            line = offset + line
            offset = ""
        # remove the new line pr space in the first sentence if exists
        if line[0] == "\n" or line[0] == " ":
            line = line[1:]
        if line[-1] == "\n" or line[-1] == " ":
            line = line[:-1]
        elif line[-1] == "[" or line[-1] == "(":
            offset = line[-1]
            line = line[:-1]
        perplexity_, _ = calc_perplexity(tokenizer, model, device, line)
        perplexity_per_line.append(perplexity_)

    burstiness = max(perplexity_per_line)
    perplexity_per_line_avg = sum(perplexity_per_line)/len(perplexity_per_line)
    msg, label, threshold = labeling(perplexity_per_line_avg)

    results = {
        "lines": lines,
        "burstiness": burstiness,
        "perplexity": perplexity,
        "perplexity_per_line_avg": perplexity_per_line_avg,
        "perplexity_per_line": perplexity_per_line,
        "msg": msg,
        "label": label,
        "threshold": threshold
    }
    return results
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model, padding_side='left')

    results = detect(args.text, tokenizer, model)
    print(results)