import json
import os

def load_from_json(path: str):
    """
        Load a JSON object from a file into a variable. If the file does not exist, an empty .json file is created at the location.
        Inputs:
            path: A relative or absolute path were the json file is stored
    """
    if not (os.path.isfile(path) and os.access(path, os.R_OK)):
        data = {}
        save_as_json(data, path)
    with open(path, mode='r', encoding="utf-8") as json_file:
        res = json.load(json_file)
    return res

def save_as_json(obj, path: str):
    """
        save given input as json file.
        Inputs:
            obj: a python dictionary or list that will be saved
            path: A relative or absolute path were the json file is stored
    """
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(obj, fp)