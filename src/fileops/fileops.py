"""
    Saving and loading operations to access data from filesystem.
    Specialized methods for this projects JSON structures.
"""
import json
import os
import numpy as np

def load_from_json(path: str):
    """
        Load a JSON object from a file into a variable.
        If the file does not exist, an empty .json file is created at the location.
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

def get_all_instances(path)->list:
    """
        From a given path get all tsp instances as list of dictionaries
    """
    res = []
    filenames = [filename for filename in os.listdir(path)
         if os.path.isfile(os.path.join(path, filename))]

    filenames.sort(key = lambda name: int(name.split('_')[-2]))
    for filename in filenames:
        res.append(load_from_json(path+filename))
    return res

def get_all_heuristics(path)->dict:
    """
        From a given path get all heuristic like data as dictionary
    """
    res = {}
    filenames = [filename for filename in os.listdir(path)
             if os.path.isfile(os.path.join(path, filename))]

    for filename in filenames:
        res[filename.replace('.json', '')] = np.array(
            load_from_json(path + filename)
        )
    return res
