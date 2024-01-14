import json
import os

def load_from_json(path: str):
    """Load a JSON object from a file into a variable"""
    if not (os.path.isfile(path) and os.access(path, os.R_OK)):
        data = {}
        save_as_json(data, path)
    with open(path, mode='r', encoding="utf-8") as json_file:
        res = json.load(json_file)
    return res

def save_as_json(obj, path: str):
    """
    save given input as json file
    """
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(obj, fp)