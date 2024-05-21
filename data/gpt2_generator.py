import argparse 
import json
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURR_DIR)

dataset_dct = {
    "tofel": "".join(PARENT_DIR+"/dataset/human/tofel.json"),
    "arxiv": "".join(PARENT_DIR+"/dataset/human/arxiv.json"),
    "essay": "".join(PARENT_DIR+"/dataset/human/student_essay.json"),
    "essay_cs": "".join(PARENT_DIR+"/dataset/human/student_cs_essay.json"),
}

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label

def complete_text(model, tokenizer, data: list, device):
    json_lst = []
    
    # encodes
    encoded_inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
    encoded_inputs = encoded_inputs.to(device)

    # generate
    output = model.generate(
        input_ids=encoded_inputs.input_ids[:, :30],
        attention_mask=encoded_inputs.attention_mask[:, :30],
        pad_token_id=tokenizer.pad_token_id,
        max_length=200,
        return_dict_in_generate=True
    )

    # decode
    sequences_list = output.sequences.tolist()
    decoded_inputs = tokenizer.batch_decode(sequences_list, skip_special_tokens=True)
    for decoded in decoded_inputs:
        json_lst.append({"input": decoded, "label": "ai"})
    return json_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--file_name", type=str, default="test.json")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.dataset in dataset_dct:
        with open(dataset_dct[args.dataset], "r") as f:
            dataset = json.load(f)
            dataset = CustomDataset([item['input'] for item in dataset], 
                        [item['label'] for item in dataset])
            dataloader = DataLoader(dataset, batch_size=args.batch_size)

    else:
        print("wrong dataset parameter")

    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    json_lst = []

    for idx, data in enumerate(dataloader):
        print(f"{idx+1} batch finished")
        json_lst.extend(complete_text(model, tokenizer, [item for item in data[0]], device))

    # save as json file
    file_path = "".join(PARENT_DIR+'/dataset/ai/'+args.file_name)
    print(f"saved to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(json_lst, f, indent=4)



