import argparse 
import re
import math
import numpy as np
import itertools
import torch
from multiprocessing.pool import ThreadPool
from scipy.stats import norm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, AutoModelForSeq2SeqLM
import os

# Set TOKENIZERS_PARALLELISM to either 'true' or 'false'
# os.environ["TOKENIZERS_PARALLELISM"] = "true"  # or "false"


threshold = 0.7
chunk_value = 100
device = "cpu" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def random_word_mask(text, ratio):
    span = 2
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = ratio//(span + 2)
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span)
        end = start + span
        search_start = max(0, start - 1)
        search_end = min(len(tokens), end + 1)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text, n_masks

def unmasker(text, num_of_masks, t5_tokenizer, t5_model):
    num_of_masks = max(num_of_masks)
    stop_id = t5_tokenizer.encode(f"<extra_id_{num_of_masks}>")[0]
    tokens = t5_tokenizer(text, return_tensors="pt", padding=True)
    for key in tokens:
        tokens[key] = tokens[key].to(device)

    output_sequences = t5_model.generate(**tokens, max_length=512, do_sample=True, top_p=0.96, num_return_sequences=1, eos_token_id=stop_id)
    results = t5_tokenizer.batch_decode(output_sequences, skip_special_tokens=False)

    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in results]
    pattern = re.compile("<extra_id_\d+>")
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    # apply extracted fills
    perturbed_texts = []
    for idx, (text_, fills) in enumerate(zip(text, extracted_fills)):
        tokens = list(re.finditer("<extra_id_\d+>", text_))
        if len(fills) < len(tokens):
            continue

        offset = 0
        for fill_idx in range(len(tokens)):
            start, end = tokens[fill_idx].span()
            text_ = text_[:start+offset] + fills[fill_idx] + text_[end+offset:]
            offset = offset - (end - start) + len(fills[fill_idx])
        perturbed_texts.append(text_)
    return perturbed_texts


def generate_text_perturb(args):
    original_text, n = args[0], args[1]
    t5_tokenizer, t5_model = args[2], args[3]
    texts = list(re.finditer("[^\d\W]+", original_text))
    ratio = int(0.3 * len(texts))

    # mask random word
    # mask_texts, list_num_of_masks = self.multiMaskRandomWord(original_text, ratio, n)
    mask_texts = []
    list_num_of_masks = []
    for i in range(n):
        mask_text, num_of_masks = random_word_mask(original_text, ratio)
        mask_texts.append(mask_text)
        list_num_of_masks.append(num_of_masks)

    # replace mask
    # list_generated_sentences = self.replaceMask(mask_texts, list_num_of_masks)
    with torch.no_grad():
        list_generated_sentences = unmasker(mask_texts, list_num_of_masks, t5_tokenizer, t5_model)
    return list_generated_sentences


def perturb(original_text, t5_tokenizer, t5_model, n, remaining=100):
    """
    original_text: string representing the sentence
    n: top n mask-filling to be choosen
    remaining: The remaining slots to be fill
    """

    if remaining <= 0:
        return []

    # torch.manual_seed(0)
    # np.random.seed(0)
    out_sentences = []
    pool = ThreadPool(remaining//n)
    out_sentences = pool.map(generate_text_perturb, [(original_text, n, t5_tokenizer, t5_model) for _ in range(remaining//n)])
    out_sentences = list(itertools.chain.from_iterable(out_sentences))
    return out_sentences

def calc_log_likelihood(sentence, tokenizer, model):
    """ 
    calculate log likelihood given text
    """
    stride = 512
    max_length = model.config.n_positions

    encodings = tokenizer(sentence, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return -1 * torch.stack(nlls).sum() / end_loc


def get_score(line, tokenizer, model, t5_tokenizer, t5_model):
    """
    Given a sentence, perturb the sentence and score
    """
    original_sentence = line
    sentence_length = len(list(re.finditer("[^\d\W]+", line)))
    # remaining = int(min(max(100, sentence_length * 1/9), 200))
    remaining = 50
    # sentences = self.mask(original_sentence, original_sentence, n=50, remaining=remaining)
    
    generated_sentences = perturb(original_sentence, t5_tokenizer, t5_model, n=50, remaining=remaining)

    real_log_likelihood = calc_log_likelihood(original_sentence, tokenizer, model)

    generated_log_likelihoods = []
    # for sentence in sentences:
    for sentence in generated_sentences:
        generated_log_likelihoods.append(calc_log_likelihood(sentence, tokenizer, model).cpu().detach().numpy())

    if len(generated_log_likelihoods) == 0:
        return -1

    generated_log_likelihoods = np.asarray(generated_log_likelihoods)
    mean_generated_log_likelihood = np.mean(generated_log_likelihoods)
    std_generated_log_likelihood = np.std(generated_log_likelihoods)

    diff = real_log_likelihood - mean_generated_log_likelihood

    score = diff/(std_generated_log_likelihood)

    return float(score), float(diff), float(std_generated_log_likelihood)


def labeling(mean_score) -> tuple:
    if mean_score >= threshold:
        return "This text is most likely generated by an A.I.", 0, threshold 
    else:   
        return "This text is most likely written by an Human", 1, threshold

def detect(text, tokenizer=None, model=None, t5_tokenizer=None, t5_model=None) -> dict:   
    threshold = 0.7
    chunk_value = 100

    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if model is None:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    if t5_tokenizer is None:
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=512, legacy=False)
    if t5_model is None:
        t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

    model = model.to(device)
    t5_model = t5_model.to(device)


    text = re.sub("\[[0-9]+\]", "", text) # remove all the [numbers] cause of wiki
    words = re.split("[ \n]", text)

    groups = len(words) // chunk_value + 1
    lines = []
    stride = len(words) // groups + 1
    for i in range(0, len(words), stride):
        start_pos = i
        end_pos = min(i+stride, len(words))

        selected_text = " ".join(words[start_pos:end_pos])
        selected_text = selected_text.strip()
        if selected_text == "":
            continue
        
        lines.append(selected_text)

    
    # sentence by sentence
    offset = ""
    scores = []
    probs = []
    final_lines = []
    labels = []
    for line in lines:
        if re.search("[a-zA-Z0-9]+", line) == None:
            continue
        score, diff, sd = get_score(line, tokenizer, model, t5_tokenizer, t5_model)
        torch.mps.empty_cache() if device == "mps" else torch.cuda.empty_cache() if device == "cuda" else None
        print("done!!!!")
        if score == -1 or math.isnan(score):
            continue
        scores.append(score)

        final_lines.append(line)
        if score >= threshold:
            labels.append(1)
            prob = "{:.2f}%\n(A.I.)".format(norm.cdf(abs(threshold - score)) * 100)
            probs.append(prob)
        else:
            labels.append(0)
            prob = "{:.2f}%\n(Human)".format(norm.cdf(abs(threshold - score)) * 100)
            probs.append(prob)

    mean_score = sum(scores)/len(scores)

    mean_prob = norm.cdf(abs(threshold - mean_score)) * 100
    msg, label, threshold = labeling(mean_score)

    results = {
        "mean_prob": mean_prob,
        "mean_score": mean_score,
        "lines": "",
        "msg": msg, 
        "label": label,
        "threshold": threshold,
    }
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model, padding_side='left')
    model = GPT2LMHeadModel.from_pretrained(args.model)

    t5_tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=512)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    
    results = detect(args.text, tokenizer, model, t5_tokenizer, t5_model)
    print(results)
