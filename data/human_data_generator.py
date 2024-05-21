import argparse 
import json
import os
from datasets import load_dataset

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURR_DIR)

def load_tofel():
    # read tofel 91 data
    file_path = "".join(PARENT_DIR+"/dataset/human/original_tofel.json")
    with open(file_path, "r") as f:
        json_lst = json.load(f)

    for item in json_lst:
        item['input'] = item['document']
        item['label'] = 'human'
        del item['document']

    # save as json file
    file_path = "".join(PARENT_DIR+"/dataset/human/tofel.json")
    print(f"saved to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(json_lst, f, indent=4)

def load_arxiv(n_dataset):
    raw_datasets = load_dataset("gfissore/arxiv-abstracts-2021", streaming=True)

    json_lst = []
    for idx, item in enumerate(raw_datasets['train']):
        if idx >= n_dataset:
            break
        json_lst.append({"input": item['abstract'].strip(), "label": "human"})

    # save as json file
    file_path = "".join(PARENT_DIR+"/dataset/human/arxiv.json")
    print(f"saved to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(json_lst, f, indent=4)

def load_student_cs_essay(n_dataset):
    raw_datasets = load_dataset("qwedsacf/ivypanda-essays", streaming=True)

    json_lst = []
    for idx, item in enumerate(raw_datasets['train']):
        if item['SOURCE'].lower().__contains__('computer'):
            json_lst.append({"input": item['TEXT'], "label": "human"})

    # save as json file
    file_path = "".join(PARENT_DIR+"/dataset/human/student_cs_essay.json")
    print(f"saved to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(json_lst, f, indent=4)


def load_student_essay(n_dataset):
    raw_datasets = load_dataset("qwedsacf/ivypanda-essays", streaming=True)

    json_lst = []
    for idx, item in enumerate(raw_datasets['train']):
        if idx >= n_dataset:
            break
        json_lst.append({"input": item['TEXT'], "label": "human"})

    # save as json file
    file_path = "".join(PARENT_DIR+"/dataset/human/student_essay.json")
    print(f"saved to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(json_lst, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tofel", type=bool, default=False)
    parser.add_argument("--arxiv", type=bool, default=False)
    parser.add_argument("--student_essay", type=bool, default=False)
    parser.add_argument("--student_cs_essay", type=bool, default=False)
    parser.add_argument("--n_dataset", type=int, default=1000)
    args = parser.parse_args()

    if args.tofel:
        load_tofel()

    if args.arxiv:
        load_arxiv(args.n_dataset) 

    if args.student_essay:
        load_student_essay(args.n_dataset)

    if args.student_cs_essay:
        load_student_cs_essay(args.n_dataset)

