
import argparse 
import json
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURR_DIR)

_ = load_dotenv()

dataset_dct = {
    "tofel": "".join(PARENT_DIR+"/dataset/human/tofel.json"),
    "arxiv": "".join(PARENT_DIR+"/dataset/human/arxiv.json"),
    "essay": "".join(PARENT_DIR+"/dataset/human/student_essay.json"),
    "essay_cs": "".join(PARENT_DIR+"/dataset/human/student_cs_essay.json"),
}


model_dct = {
    "llama3_8b": "Llama3-8b-8192",
    "llama3_70b": "Llama3-70b-8192",
}

prompt_template_lst = [
    'Enhance the word choices to sound more like that of a native speaker',
    'Simplify word choices as if written by a non-native speaker',
    'Help me polish this',
    'Rewrite this for me',
    'Refine this for me please',
]

def check_triple_backticks(input):
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, input, re.DOTALL)

    str_ = ""
    if matches:
        for match in matches:
            str_ += match.strip()
    else:
        print("No matches found.")

    return str_

def paraphrase(model, data: dict, prompt_template: str):
    prompt = """# task: {prompt_template}. Follow output format.

    # input: 
    ```{input}```

    # output format: 
    ```your answer```
    """.format(prompt_template=prompt_template, input=data['input'])
    
    result = model.invoke(prompt).content
    result = check_triple_backticks(result)
    return result        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--model", type=str, default="llama3_8b")
    parser.add_argument("--idx_prompt_template", type=int, default=0)
    parser.add_argument("--file_name", type=str, default="test.json")
    parser.add_argument("--n_dataset", type=int, default=0)
    args = parser.parse_args()

    if args.dataset in dataset_dct:
        with open(dataset_dct[args.dataset], "r") as f:
            dataset = json.load(f)
    else:
        print("wrong dataset parameter")

    if args.model in model_dct:
        model = ChatGroq(model=model_dct[args.model])
    else:
        print("wrong model parameter")


    prompt_template = prompt_template_lst[args.idx_prompt_template]
    json_lst = []

    for data in dataset[:]:
        result_ = paraphrase(model, data, prompt_template)
        json_lst.append({"input": result_.strip(), "label": "ai"})

    # save as json file
    file_path = "".join(PARENT_DIR+'/dataset/ai/'+args.file_name)
    print(f"saved to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(json_lst, f, indent=4)