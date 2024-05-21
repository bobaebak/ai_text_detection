import re 
import json

def extract_only_answers(key, raw_text):
    """
    Extracts only the answers from the given raw text.

    Args:
    - key (str): A unique identifier/key for the extracted answers.
    - raw_text (str): The raw text containing the answers.

    Returns:
    - list: A list of dictionaries containing the extracted answers and their labels.
            The label is set to "human" for all extracted answers.

    Example:
    >>> raw_text = "Given Answer: This is a sample answer. Correct Response Feedback: ..."
    >>> extract_only_answers("input", raw_text)
    [{'input': 'This is a sample answer.', 'label': 'human'}]
    """
    lst = []

    given_answer_pattern = r"Given Answer:\s*([\s\S]*?)(Correct|Incorrect|Partial Credit)*(?=\s*Response Feedback:|\Z)"

    # Extract the text following "Given Answer:" and its correctness using regex
    given_answers = re.findall(given_answer_pattern, raw_text)

    # Check if the word exists in each "Given Answer" section and print its correctness
    if given_answers:
        idx = 0
        for given_answer in given_answers:
            st_answer = given_answer[0].strip()
            if st_answer != '':
                lst.append({f"{key}": st_answer, "label": "human"})
                # print(given_answer[0].strip())
                idx += 1
    else:
        print("Given Answer not found in the text.")
    
    return lst

# load a raw text file
with open('../dataset/human/student_answers.txt', "r") as f:
    text = f.read()  

# extract only student answer
json_lst = extract_only_answers("input", text)

# save as json file
with open('../dataset/human/eval_human.json', "w") as f:
    json.dump(json_lst, f, indent=4)
