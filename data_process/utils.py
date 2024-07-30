from sklearn.decomposition import PCA
import numpy as np
from sentence_transformers import SentenceTransformer
import random
import torch
import json
import matplotlib.pyplot as plt
import re
import copy
import re
from typing import List, Callable, Dict


first_level_method_list = ["GetObsImage", "GetPromptImages", "GetSceneImage", "GetPromptAssets", "GetWholeTask"]

second_level_method_list = ['generate', 'GetAllObjectsFromImage', 'GetAllObjectsFromPromptImage', 'GetAllSameTextureObjects', 'GetAllSameShapeObjects', 'GetSameShapeObjectsAsObject', 'GetObjectsWtihGivenTexture', 'GetDragObjName', 'GetBaseObjName', 'PickPlace', 'DistractorActions', 'RearrangeActions', 'SelectFromScene', 'GetStepsLocForObject', 'GetAllObjextsStepsLoc', 'SelectObj', 'RotateAll', 'RecDegree', 'RecAdj', 'PlanOrder', 'CalculateValidArea', 'MultiPPWithConstrain', 'MultiPPWithSame', 'GetReverse', 'Sorted', 'GetSweptAndBoundaryName', 'GetTimes', 'RobotExecution']

SENTENCEBERT_PATH = "/root/models/all-mpnet-base-v2"
random.seed(3072)
device_id = 0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#FF5733', '#33FF57', '#5733FF', '#FF336E', '#33FFC4', '#FFD733', '#FF3333']
labels = ['Put', 'scene', 'Rotate', 'Rearrange', 'Rearrange_then', 
          'novel_adj', 'novel_name', 'novel_a_n', 'Twist', 'motion','Stack','exceeding',
          'touching','texture','shape','First','order']


def get_all_vima_prompt():
    with open("data/vima_prompt.txt","r") as f:
        vima_prompt_list = f.readlines()
    return vima_prompt_list


def get_all_vima_instruction():
    with open("data/vima_instruction1.txt","r") as f:
        vima_prompt_list = f.readlines()
    return vima_prompt_list

def load_sentence_bert(path):
    model = SentenceTransformer(path)
    model.to(torch.device(f"cuda:{device_id}"))
    return model


def get_sentence_bert_embedding(data):
    model = load_sentence_bert(SENTENCEBERT_PATH)
    embeddings = model.encode(data)
    return embeddings
    

def get_instruction(codebase):
    instruction_list = []
    for i in range(len(codebase)):
        instruction_list.append(codebase[i]["instruction"])
    return instruction_list

def get_code(codebase):
    code_list = []
    for i in range(len(codebase)):
        code_list.append(codebase[i]["template"])
    return code_list

def save_result_to_json(path,result):
    with open(path,"w") as f:
        json.dump(result,f,indent=4)

def get_and_save_final_result(final_path,total_path):
    result_dict = {}

    value_list = []
    with open(final_path,"r") as f:
        data = json.load(f)
    for key,value in data.items():
        value_list.append(value)
    result_dict = value_list[0].copy()

    for dictionary in value_list[1:]:
        for key, value in dictionary.items():
            result_dict[key] += value
    for key in result_dict:
        result_dict[key] /= 17

    save_result_to_json(total_path,result_dict)


def extract_template_with_number(gpt_result):
    template_list = []
    matches = re.findall(r'\d+\.\s*([^\.]+)\.', gpt_result)
    for match in matches:
        template_list.append(match.strip())
    return template_list

def insert_template(template,instruction,replace_string):
    return template.replace(replace_string,instruction)

def read_txt(path):
    with open(path,"r") as f:
        data = f.readlines()
    return data

def read_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

def get_one_specific_str_index():
    pass

def get_all_specific_str_index(index_str,input_string):
    indices = []
    start_index = 0

    while True:
        index = input_string.find(index_str, start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1

    return indices

def analysis_gpt4_code(result: str, task_name: str) -> List[str]:
    codes = [code.strip() for code in result.splitlines() if not code.strip().startswith("#")]
    
    task_handlers: Dict[str, Callable[[List[str]], List[str]]] = {
        "visual_manipulation": handle_visual_manipulation,
        "rotate": handle_rotate,
        "rearrange": handle_rearrange,
        "rearrange_then_restore": handle_rearrange,
        "pick_in_order_then_restore": handle_pick_in_order,
        "follow_motion": handle_follow_motion,
        "follow_order": handle_follow_order,
        "scene_understanding": handle_scene_understanding,
        "same_texture": handle_same_texture,
        "same_shape": handle_same_shape,
        "novel_noun": handle_novel_noun,
        "manipulate_old_neighbor": handle_manipulate_old_neighbor,
        "twist": handle_twist,
        "sweep_without_exceeding": handle_sweep_without_exceeding,
        "novel_adj": handle_novel_adj,
        "novel_adj_and_noun": handle_novel_adj
    }
    
    handler = task_handlers.get(task_name)
    if not handler:
        raise ValueError(f"No such task: {task_name}")
    
    return handler(codes)

def is_assignment(code: str) -> bool:
    if not code or " = " not in code:
        return False
    parts = code.split("= ", 1)
    if len(parts) < 2:
        return False
    value = parts[1].strip()
    return (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'"))

def truncate_annotation(code: str) -> str:
    return code.split("#", 1)[0].strip() if "#" in code else code

def is_digit(code: str) -> bool:
    if not code or " = " not in code:
        return False
    try:
        int(code.split("= ", 1)[1])
        return True
    except ValueError:
        return False

def handle_visual_manipulation(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "top_image", "MMPROBO.generate", "PickPlace"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_rotate(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        code = truncate_annotation(code)
        if is_assignment(code) or is_digit(code) or any(keyword in code for keyword in ["GetObsImage", "top_image", "MMPROBO.generate", "PickPlace"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_rearrange(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        code = truncate_annotation(code)
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "top_image", "GetSceneImage", "MMPROBO.generate", "DistractorActions", "GetAllObjectsFromImage", "GetAllObjectsFromPromptImage", "RearrangeActions"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_pick_in_order(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "MMPROBO.generate", "PickPlace", "original_container_loc = "]):
            return_code.append(code)
    return return_code

def handle_follow_motion(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "GetPromptImages", "GetStepsLoc", "PickPlace", "query = "]):
            return_code.append(code)
    return return_code

def handle_follow_order(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "GetPromptImages", "GetAllObjextsStepsLoc"]):
            return_code.append(code)
        elif "PlanOrder" in code:
            full_code = code
            for j in range(1, 6):
                if ")" in codes[j]:
                    full_code += codes[j]
                    break
                full_code += codes[j]
            return_code.append(full_code)
            break
        elif "return info" in code:
            break
    return return_code

def handle_scene_understanding(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "GetSceneImage", "top_image", "GetAllObjectsFromImage", "GetAllObjectsFromPromptImage", "SelectFromScene", "MMPROBO.generate", "PickPlace"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_same_texture(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "GetAllSameTextureObjects", "MultiPPWithSame", "MMPROBO.generate"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_same_shape(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if code and (is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "query = ", "GetAllSameProfileObjects", "MultiPPWithSame", "MMPROBO.generate"])):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_novel_noun(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "GetPromptAssets", "MMPROBO.generate", "PickPlace", "GetDragObjName", "GetBaseObjName"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_manipulate_old_neighbor(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "MMPROBO.generate", "PickPlace"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_twist(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in ["GetObsImage", "GetPromptImages", "GetObjectsWtihGivenTexture", "RecDegree", "RotateAll"]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_sweep_without_exceeding(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in [
            "GetObsImage", "GetPromptAssets", "GetSweptAndBoundaryName", "GetSweptObjName",
            "GetBoundsObjName", "GetTimes", "GetWholeTask", "GetConstraintObjName",
            "MMPROBO.generate", "CalculateValidArea", "valid_area", "sorted", "MultiPPWithConstrain"
        ]):
            return_code.append(code)
        elif "return info" in code:
            break
    return return_code

def handle_novel_adj(codes: List[str]) -> List[str]:
    return_code = []
    for code in codes:
        if is_assignment(code) or any(keyword in code for keyword in [
            "front_image =", "GetObsImage", "GetPromptImages", "GetPromptAssets", "GetWholeTask",
            "GetReverse", "is_reverse", "RecAdj", "GetDragObjName", "GetSameShapeObjectsAsObject",
            "MMPROBO.generate", "SelectObj", "GetBaseObjName", "PickPlace"
        ]):
            return_code.append(code)
        elif ")" in code and "(" not in code:
            return_code[-1] = return_code[-1] + code.strip()
        elif "return info" in code:
            break
    return return_code
    
def extract_numbers(text):
    pattern = r'\d+'
    numbers = re.findall(pattern, text)
    return numbers[0]

def get_direction_of_origin_prompt(prompt):
    if "east" in prompt:
        return "east"
    elif "west" in prompt:
        return "west"
    elif "north" in prompt:
        return "north"
    else :
        return "south"
    
def get_adj_of_origin_prompt(prompt):
    if "daxer" in prompt:
        return "daxer"
    elif "blicker" in prompt:
        return "blicker"
    elif "modier" in prompt:
        return "modier"
    else :
        return "kobar"
    
def get_twist_texture(prompt):
    index_all = prompt.find("all ")
    index_objects = prompt.find(" objects")
    substring_between = prompt[index_all + len("all "):index_objects].strip()
    return substring_between

def get_scene_two_object(prompt):
    pattern_the = r'\bthe\b'
    pattern_object = r'\bobject\b'
    matches_the = [(m.start(0), m.end(0)) for m in re.finditer(pattern_the, prompt)]

    matches_object = [(m.start(0), m.end(0)) for m in re.finditer(pattern_object, prompt)]

    print("Indexes of 'the' occurrences:")
    for match in matches_the:
        print(match)

    print("\nIndexes of 'object' occurrences:")
    for match in matches_object:
        print(match)
    return prompt[matches_the[0][0]:matches_object[0][0] + 6], prompt[matches_the[1][0]:matches_object[1][0] + 6]

def change_number_in_sweep(prompt):
    number = prompt.split(" ")[1]
    return number

def find_max_and_second_max_index(arr):
    sorted_indices = sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)
    return sorted_indices[:2]

def insert_task_into_prompt(task, prompt_base, insert_index="INSERT TASK HERE"):
    full_prompt = prompt_base.replace(insert_index, task)
    return full_prompt





    





