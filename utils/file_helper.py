import json 

def load_json_list(file_name: str) -> list:
    with open(file_name, "r") as f:
        json_list = json.load(f)

    return json_list

def save_json_list(file_name: str, json_list: list):
    with open(file_name, "w") as f:
        json.dump(json_list, f, indent=4)
    
    print("JSON array saved to", file_name)