"""
Using prompt, paraphrase them
"""

import argparse 
import json 
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURR_DIR)

_ = load_dotenv() 

dataset_dct = {
    "tofel": "".join(PARENT_DIR+"/dataset/human/tofel.json"),
    "arxiv": "".join(PARENT_DIR+"/dataset/human/arxiv.json"),
    "essay": "".join(PARENT_DIR+"/dataset/human/student_essay.json"),
    "essay_cs": "".join(PARENT_DIR+"/dataset/human/student_cs_essay.json"),
    "eval": "".join(PARENT_DIR+"/dataset/human/eval_human.json"),
}

prompt_template_lst = [
    'Enhance the word choices to sound more like that of a native speaker',
    'Simplify word choices as if written by a non-native speaker',
    'Help me polish this',
    'Rewrite this for me',
    'Refine this for me please',
]

def paraphrase(model=None, data: dict=None, prompt_template: str=None):
    if model is None:
        model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    if prompt_template is None:
        prompt_template = prompt_template_lst[0]
        
    prompt = """{prompt_template}: {input}""".format(prompt_template=prompt_template, input=data['input'])
    messages = [("human", prompt)]
    result = model.invoke(messages).content
    return result
    

# prompt_template = "Enhance the word choices to sound more like that of a native speaker: "

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--idx_prompt_template", type=int, default=0)
    parser.add_argument("--file_name", type=str, default="test.json")
    args = parser.parse_args()
    
    if args.dataset in dataset_dct:
        with open(dataset_dct[args.dataset], "r") as f:
            dataset = json.load(f)
    else:
        print("wrong dataset parameter")

    model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    
    prompt_template = prompt_template_lst[args.idx_prompt_template]

    json_lst = []
    print("start!")
    for idx, data in enumerate(dataset):
        print(idx+1, " finished")
        result_ = paraphrase(model, data, prompt_template)
        json_lst.append({"input": result_.strip(), "label": "ai"})
        
    # save as json file
    file_path = "".join(PARENT_DIR+'/dataset/ai/'+args.file_name)
    print(f"saved to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(json_lst, f, indent=4)