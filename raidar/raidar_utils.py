import os
import json 
import openai
import numpy as np
from fuzzywuzzy import fuzz

NGRAM_NUM = 4
openai.api_key = "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"
client = openai.OpenAI(api_key=openai.api_key)

def load_curr_path():
    return os.getcwd()


def load_project_path():
    return os.path.abspath(os.path.join(load_curr_path(), os.pardir))


def load_json_list(file_name: str) -> list:
    with open(file_name, "r") as f:
        json_list = json.load(f)

    return json_list


def save_json_list(file_name: str, json_list: list):
    with open(file_name, "w") as f:
        json.dump(json_list, f, indent=4)
    
    print("JSON array saved to", file_name)


def gpt_completion(prompt_inst: str, text: str):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{
            "role": "user",
            "content": f"{prompt_inst}: {text}",
        }],
        # temperature=0
    )

    content = response.choices[0].message.content.strip()
    return response, content

def ai_answer_generator(human_json_list: list) -> list:
    """
    Generate fake answer data using human data
    """
    ai_json_list = []

    prompt_inst = "Write a very short and concise review based on this"

    for ix, answer in enumerate(human_json_list):
        _, content = gpt_completion(prompt_inst, answer['answer'])
        ai_json_list.append({'answer': content})
    
        if ix%10 == 0:
            print(f"{ix} is finished") 
    
    return ai_json_list

def rewrite_inv_text(json_list: list):
    """
    Given a input, generate rewrite text using 7 prompt templates
    """

    prompt_inst_list = [
        'Revise this with your best effort', 
        'Help me polish this', 
        'Rewrite this for me', 
        'Make this fluent while doing minimal change', 
        'Refine this for me please', 
        'Concise this for me and keep all the information',
        'Improve this in GPT way'
    ]

    rewrite_list = []
    for ix, data in enumerate(json_list):
        tmp_dct = {}
        tmp_dct['input'] = data['answer'] # original text
        
        for prompt_inst in prompt_inst_list:
            _, content = gpt_completion(prompt_inst, data['answer'])
            tmp_dct[prompt_inst] = content 
        
        rewrite_list.append(tmp_dct)

        print(f"{ix} is finished")
    
    return rewrite_list


def rewrite_equi_text(json_list: list) -> list:
    """
    Given a input, generate rewrite text using reverse, reverse transport
    """
    
    prompt_inst_list = [
        {"Transformed": 'Rewrite to Expand this', "Reversal": 'Rewrite to Concise this'},
        {"Transformed": 'Write this in the opposite tone', "Reversal": 'Write this in the opposite tone'},
        {"Transformed": 'Rewrite this in the opposite meaning', "Reversal": 'Rewrite this in the opposite meaning'},
    ]
    
    rewrite_list = []
    
    print(f"#### Run the Equiavance rewrite on {len(json_list)}")
    
    for ix, data in enumerate(json_list):
        tmp_dct = {}
        tmp_dct['input'] = data['answer']

        for prompt_inst in prompt_inst_list:
            prompt_inst_transformed = prompt_inst['Transformed']
            prompt_inst_reversal = prompt_inst['Reversal']

            _, answer1 = gpt_completion(prompt_inst_transformed, data['answer'])
            tmp_dct['tmp&_' + prompt_inst_transformed] = answer1

            _, answer2 = gpt_completion(prompt_inst_reversal, answer1)
            tmp_dct['final*_' + prompt_inst_reversal] = answer2
        
        rewrite_list.append(tmp_dct)

        print(f"{ix} is finished")

    return rewrite_list

def select_edit_method(name):
    edit_handler_types = {
        "bow": Bow,
        "lev": Lev 
    }
    
    cls = edit_handler_types.get(name)
    return cls

def inv_data_generate(json_list: list) -> list:
    """
    Generate Invariance Data
    """
    total_len = len(json_list)

    for data in json_list:
        original = data['input']

        bow_dct = {}
        all_bow_dct = [0 for i in range(NGRAM_NUM)]
        cnt = 0        

        lev_dct = {}
        whole_combined=''
        
        for prompt_inst in data.keys():
            if prompt_inst != 'common_features':
                whole_combined += (' ' + data[prompt_inst])

                # compute bag of words across the n-grams
                bow_cls = select_edit_method('bow')(original, data[prompt_inst])
                b_res = bow_cls.calculate_sentence_common()
                bow_dct[prompt_inst] = b_res 
                all_bow_dct = bow_cls.sum_for_list(all_bow_dct, b_res)

                # compute levenshtein distance 
                lev_cls = select_edit_method('lev')(original, data[prompt_inst])
                l_res = lev_cls.calculate_lev_distance()
                lev_dct[prompt_inst] = l_res 

                cnt += 1
        
        data['fzwz_features'] = lev_dct
        data['common_features'] = bow_dct
        data['avg_common_features'] = [a/cnt for a in all_bow_dct]
        data['common_features_ori_vs_allcombined'] = select_edit_method('bow')(original, whole_combined).calculate_sentence_common()

    return json_list


class Bow:
    def __init__(self, text1, text2):
        self.text1 = text1 
        self.text2 = text2

    def tokenize_and_normalize(self, sentence):
        return [word.lower().strip() for word in sentence.split()]

    def extract_ngrams(self, tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def common_elements(self, list1, list2):
        # Find common elements between two lists
        return set(list1) & set(list2)

    def calculate_sentence_common(self):
        tokens1 = self.tokenize_and_normalize(self.text1)
        tokens2 = self.tokenize_and_normalize(self.text2)

        number_common_hierarchy = []

        for n in range(1, NGRAM_NUM+1):
            if n == 1:
                # Find common words
                common_words = self.common_elements(tokens1, tokens2)
                number_common_hierarchy.append(len(list(common_words)))

            else:
                ngrams1 = self.extract_ngrams(tokens1, n)
                ngrams2 = self.extract_ngrams(tokens2, n)
                common_ngrams = self.common_elements(ngrams1, ngrams2) 
                number_common_hierarchy.append(len(list(common_ngrams)))

        return number_common_hierarchy
    
    def sum_for_list(self, a,b):
        return [aa+bb for aa, bb in zip(a,b)]


class Lev:
    def __init__(self, text1, text2):
        self.text1 = text1 
        self.text2 = text2

    def calculate_lev_distance(self):
        return [fuzz.ratio(self.text1, self.text2), fuzz.token_set_ratio(self.text1, self.text2)]
    

def compare_bow_avg(human, ai):
    for i, (h_, ai_) in enumerate(zip(human, ai)):
        h_avg = np.mean(h_['avg_common_features'])
        ai_avg = np.mean(ai_['avg_common_features'])

        if h_avg <= ai_avg:
            print(i)